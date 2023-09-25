import os
import matplotlib.pyplot as plt
import numpy as np
import torch.multiprocessing as mp
from tensorboardX import SummaryWriter
from common_sr.srollout import RolloutWorker
from agents.sagent import Agents
from common_sr.replay_buffer import ReplayBuffer
import time
from tqdm import tqdm

class Runner:
    def __init__(self, env, args):
        self.env = env

        self.agents = Agents(args)

        self.rolloutWorker = RolloutWorker(env, self.agents, args)

        if not args.evaluate:
            self.buffer = ReplayBuffer(args)
        self.args = args
        self.win_rates = []
        self.episode_rewards = []

        # 用来保存plt和pkl
        self.save_path = self.args.result_dir + '/' + args.alg + args.exp_dir
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

    def run(self, num):
        plot_dir = self.args.plot_dir + self.args.exp_dir + str(self.args.num_run)
        if not os.path.exists(plot_dir):
            os.makedirs(plot_dir)
        # logger = SummaryWriter(plot_dir)
        time_steps, train_steps, evaluate_steps = 0, 0, -1
        pbar = tqdm(self.args.n_steps)
        logger = SummaryWriter(plot_dir)
        if self.args.load_model == False or self.args.load_model == True:
            while time_steps < self.args.n_steps:
                if time_steps // self.args.evaluate_cycle > evaluate_steps:
                    win_rate, episode_reward = self.evaluate()
                    # episode_reward = [i for i in [2, 3]]
                    for i, rew in enumerate(episode_reward):
                        logger.add_scalar('agent%i/mean_episode_rewards' % i,
                                          episode_reward[i],
                                          time_steps)
                    self.episode_rewards.append(episode_reward)
                    evaluate_steps += self.args.evaluate_epoch
                # 收集self.args.n_episodes个episodes
                episodes = []
                # start = time.time()
                episode_batch, _, _, steps = self.rolloutWorker.generate_episode()
                # end = time.time()
                # print(end - start, 'sample with multiprocessing:', self.args.process)
                time_steps += steps
                pbar.update(steps)
                self.buffer.store_episode(episode_batch)

                # start = time.time()
                for train_step in range(self.args.train_steps):
                    mini_batch = self.buffer.sample(min(self.buffer.current_size, self.args.batch_size))
                    if self.args.alg.find('o') > -1:
                        self.agents.train(mini_batch, train_steps, self.args.epsilon)
                    else:
                        self.agents.train(mini_batch, train_steps)
                    train_steps += 1
                # end = time.time()
                    # print(end - start, 'training')
        pbar.close()
        win_rate, episode_reward = self.evaluate()
        # print('win_rate is ', win_rate)
        self.win_rates.append(win_rate)
        self.episode_rewards.append(episode_reward)

        if self.args.load_model == False:
            self.plt(num)

    def evaluate(self):
        win_number = 0
        episode_rewards = (0, 0)  # cumulative rewards

        _, episode_rewards, win_tag, _ = self.rolloutWorker.generate_episode(evaluate=True)

        episode_rewards = [episode_rewards[i] / self.args.evaluate_epoch / self.args.process for i in range(len(episode_rewards))]
        return win_number / self.args.evaluate_epoch, episode_rewards

    def plt(self, num):

        np.save(self.save_path + '/episode_rewards_{}'.format(num), self.episode_rewards)   #
        # print(self.episode_rewards)
        # plt.close()