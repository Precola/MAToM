import numpy as np
import torch
from spikingjelly.activation_based import neuron, functional

class RolloutWorker:
    def __init__(self, env, agents, args):
        self.env = env
        self.agents = agents
        self.episode_limit = args.episode_limit
        self.n_actions = args.n_actions
        self.n_agents = args.n_agents
        self.obs_shape = args.obs_shape
        self.args = args

        self.epsilon = args.epsilon
        self.anneal_epsilon = args.anneal_epsilon
        self.min_epsilon = args.min_epsilon
        print('Init RolloutWorker')

    def generate_episode(self, episode_num=None, evaluate=False):
        # if self.args.alg == 'siql_no_rnn':
        #     from policy_sc.siql_no_rnn import SIQLUR
        #     self.policy_sc = SIQLUR(self.args)
        #     self.policy_sc.eval_snn.reset()

        if self.args.replay_dir != '' and evaluate and episode_num == 0:  # prepare for save replay of evaluation
            self.env.close()
        # Store all data
        EPISODE = dict(
                        O = [],
                        U = [],
                        R = [],
                        O_NEXT = [],
                        U_ONEHOT = [],
                        AVAIL_U = [],
                        AVAIL_U_NEXT = [],
                        PADDED = [],
                        TERMINATE = [],
        )

        NUM_EPISODES = self.args.n_episodes if evaluate==False else self.args.evaluate_epoch
        episode_num = 0 if evaluate == False else self.args.evaluate_epoch

        episode_reward = np.zeros((self.args.process, self.n_agents))

        for episode_idx in range(NUM_EPISODES):
            # Store one multiprocessing data
            o, u, r, avail_u, u_onehot, terminate, padded = [], [], [], [], [], [], []
            obs = self.env.reset()
            obs1 = obs.copy()
            obs1[:, 0], obs1[:, 1], obs1[:, 2], obs1[:, 3] = \
                obs[:, 2], obs[:, 3], obs[:, 0], obs[:, 1]
            obs_ = (obs, obs1)
            obs_ = np.stack((obs, obs1), axis=0).transpose(1, 0, 2)
            num_env = obs.shape[0]

            last_action = np.zeros((self.args.n_agents, num_env, self.args.n_actions))
            self.agents.policy.init_hidden(1, num_env)
            terminated = False
            win_tag = False
            step = 0

            # epsilon
            epsilon = 0 if evaluate else self.epsilon
            if self.args.epsilon_anneal_scale == 'episode':
                epsilon = epsilon - self.anneal_epsilon if epsilon > self.min_epsilon else epsilon

            # for each episode (include 50 steps and num_env multiprocessing)
            while not terminated and step < self.episode_limit:
                # time.sleep(0.2)
                obs = np.array(obs_)    #A perspective, B perspective
                avail_action = [1] * self.args.n_actions
                actions, avail_actions, actions_onehot = [], [], []
                last_action_mem = last_action.copy()
                for agent_id in range(self.n_agents):
                    action = self.agents.choose_action(num_env, obs[:, agent_id, :], last_action[agent_id],
                                                       agent_id, avail_action, epsilon,
                                                       last_action_mem[self.n_agents - 1 -agent_id], evaluate)
                    # generate onehot vector of th action
                    action_onehot = np.zeros((num_env, self.args.n_actions))
                    for i in range(num_env): action_onehot[i, action[0, i]] = 1
                    actions.append(action[0].cpu().numpy().tolist())    #np.int(action)
                    actions_onehot.append(action_onehot)
                    avail_actions.append(avail_action)
                    last_action[agent_id] = action_onehot

                actions = np.array(actions).transpose(1,0)                  #[num_env, num_agent](4, 2)
                obs_, reward, done, info = self.env.step(actions=actions)   #[num_env,num_agent,num_state],[num_env, num_agent],[num_env] (4, 2, 10) (4, 2)

                if self.args.load_model == True:
                    self.env.render(mode="human")
                    # print(reward)

                o.append(obs)
                u.append(np.expand_dims(actions, self.n_agents))
                u_onehot.append(actions_onehot)
                avail_u.append(avail_actions)
                r.append(np.expand_dims(reward, 2))
                terminate.append(np.expand_dims(np.array([terminated]*num_env), 1))
                padded.append(np.expand_dims(np.array([0.]*num_env), 1))
                episode_reward = episode_reward + reward
                step += 1
                if self.args.epsilon_anneal_scale == 'step':
                    epsilon = epsilon - self.anneal_epsilon if epsilon > self.min_epsilon else epsilon

            # last obs
            obs = np.array(obs_)
            o.append(obs)
            o_next = o[1:]
            o = o[:-1]
            # get avail_action for last obs，because target_q needs avail_action in training
            avail_actions = []
            for agent_id in range(self.n_agents):
                avail_action = [1] * self.args.n_actions
                avail_actions.append(avail_action)
            avail_u.append(avail_actions)
            avail_u_next = avail_u[1:]
            avail_u = avail_u[:-1]

            # if step < self.episode_limit，padding (if termined before the max steps, add data to max steps)
            for i in range(step, self.episode_limit):
                o.append(np.zeros((self.n_agents, self.obs_shape)))
                u.append(np.zeros([self.n_agents, 1]))
                r.append(np.zeros([self.n_agents, 1]))
                o_next.append(np.zeros((self.n_agents, self.obs_shape)))
                u_onehot.append(np.zeros((self.n_agents, self.n_actions)))
                avail_u.append(np.zeros((self.n_agents, self.n_actions)))
                avail_u_next.append(np.zeros((self.n_agents, self.n_actions)))
                padded.append([1.]*num_env)
                terminate.append([1.]*num_env)

            # Processing data for each episode
            EPISODE['O'].append(np.stack(o, axis=0).transpose(1, 0, 2, 3))
            EPISODE['U'].append(np.stack(u, axis=0).transpose(1, 0, 2, 3).astype(int))
            EPISODE['R'].append(np.stack(r, axis=0).transpose(1, 0, 2, 3))
            EPISODE['O_NEXT'].append(np.stack(o_next, axis=0).transpose(1, 0, 2, 3))
            EPISODE['U_ONEHOT'].append(np.stack(u_onehot, axis=0).transpose(2, 0, 1, 3))
            EPISODE['AVAIL_U'].append(np.ones(EPISODE['U_ONEHOT'][0].shape))
            EPISODE['AVAIL_U_NEXT'].append(np.ones(EPISODE['U_ONEHOT'][0].shape))
            EPISODE['PADDED'].append(np.stack(padded, axis=0).transpose(1, 0, 2))
            EPISODE['TERMINATE'].append(np.stack(terminate, axis=0).transpose(1, 0, 2))

        episode_reward = episode_reward.sum(0)

        for i in EPISODE.keys():
            EPISODE[i] = np.concatenate(EPISODE[i], axis=0)
        step = step * self.args.n_episodes * num_env

        if not evaluate:
            self.epsilon = epsilon
        if evaluate and episode_num == self.args.evaluate_epoch and self.args.replay_dir != '':
            self.env.save_replay()
            self.env.close()
        return EPISODE, episode_reward, win_tag, step

    def generate_episode_sample(self, episodes, steps, episode_num=None, evaluate=False):
        if self.args.replay_dir != '' and evaluate and episode_num == 0:  # prepare for save replay of evaluation
            self.env.close()
        o, u, r, avail_u, u_onehot, terminate, padded = [], [], [], [], [], [], []
        obs = self.env.reset()
        obs_ = (obs, self.env.game._flip_coord_observation_perspective(obs))  # A perspective, B perspective
        terminated = False
        win_tag = False
        step = 0
        episode_reward = (0, 0)  # cumulative rewards

        # ###
        # for param_sc in self.agents_sc.policy_base.parameters():
        #     param_sc.requires_grad = False
        # self.agents_sc.policy_base.eval()

        last_action = np.zeros((self.args.n_agents, self.args.n_actions))
        self.agents.policy.init_hidden(1)

        # epsilon
        epsilon = 0 if evaluate else self.epsilon
        if self.args.epsilon_anneal_scale == 'episode':
            epsilon = epsilon - self.anneal_epsilon if epsilon > self.min_epsilon else epsilon

        while not terminated and step < self.episode_limit:
            # time.sleep(0.2)
            obs = np.array(obs_)    #A perspective, B perspective
            avail_action = [1] * self.args.n_actions
            actions, avail_actions, actions_onehot = [], [], []
            for agent_id in range(self.n_agents):
                action = self.agents.choose_action(obs[agent_id], last_action[agent_id], agent_id,
                                                       avail_action, epsilon, evaluate)
                # generate onehot vector of th action
                action_onehot = np.zeros(self.args.n_actions)
                action_onehot[action] = 1
                actions.append(np.int(action))
                actions_onehot.append(action_onehot)
                avail_actions.append(avail_action)
                last_action[agent_id] = action_onehot

            obs_, reward, done, info = self.env.step(actions=actions)
            # print(actions,reward)
            win_tag = True if terminated else False
            # save obs, actions, avail_actions, reward at time t
            o.append(obs)
            u.append(np.reshape(actions, [self.n_agents, 1]))
            u_onehot.append(actions_onehot)
            avail_u.append(avail_actions)
            r.append(np.reshape(reward, [self.n_agents, 1]))    #reward
            terminate.append([terminated])
            padded.append([0.])
            # episode_reward += reward
            episode_reward = [episode_reward[i] + reward[i] for i in range(min(len(episode_reward), len(reward)))]

            step += 1
            if self.args.epsilon_anneal_scale == 'step':
                epsilon = epsilon - self.anneal_epsilon if epsilon > self.min_epsilon else epsilon

            if self.args.load_model == True:
                self.env.render(mode="human")

        # last obs
        obs = np.array(obs_)
        o.append(obs)
        o_next = o[1:]
        o = o[:-1]
        # get avail_action for last obs，because target_q needs avail_action in training
        avail_actions = []
        for agent_id in range(self.n_agents):
            avail_action = [1] * self.args.n_actions
            avail_actions.append(avail_action)
        avail_u.append(avail_actions)
        avail_u_next = avail_u[1:]
        avail_u = avail_u[:-1]

        # if step < self.episode_limit，padding
        for i in range(step, self.episode_limit):
            o.append(np.zeros((self.n_agents, self.obs_shape)))
            u.append(np.zeros([self.n_agents, 1]))
            r.append(np.zeros([self.n_agents, 1]))
            o_next.append(np.zeros((self.n_agents, self.obs_shape)))
            u_onehot.append(np.zeros((self.n_agents, self.n_actions)))
            avail_u.append(np.zeros((self.n_agents, self.n_actions)))
            avail_u_next.append(np.zeros((self.n_agents, self.n_actions)))
            padded.append([1.])
            terminate.append([1.])

        episode = dict(o=o.copy(),
                       u=u.copy(),
                       r=r.copy(),
                       avail_u=avail_u.copy(),
                       o_next=o_next.copy(),
                       avail_u_next=avail_u_next.copy(),
                       u_onehot=u_onehot.copy(),
                       padded=padded.copy(),
                       terminated=terminate.copy()
                       )

        episodes[episode_num] = episode
        steps[episode_num] = step

        # add episode dim
        for key in episode.keys():
            episode[key] = np.array([episode[key]])
        if not evaluate:
            self.epsilon = epsilon
        if evaluate and episode_num == self.args.evaluate_epoch - 1 and self.args.replay_dir != '':
            self.env.save_replay()
            self.env.close()
       # return episode, episode_reward, win_tag, step


class RolloutWorker_AC:
    def __init__(self, env, args, ppo):
        self.env = env
        self.episode_limit = args.episode_limit
        self.n_actions = args.n_actions
        self.n_agents = args.n_agents
        self.obs_shape = args.obs_shape
        self.args = args
        from network.spiking_net import ActorCritic_rnn
        from spikingjelly.activation_based import neuron, functional
        # self.ppo = ActorCritic_rnn(self.obs_shape, args)
        self.ppo = ppo

        self.epsilon = args.epsilon
        self.anneal_epsilon = args.anneal_epsilon
        self.min_epsilon = args.min_epsilon
        print('Init RolloutWorker_SNN')

    def generate_episode(self, episode_num=None, evaluate=False):
        if self.args.replay_dir != '' and evaluate and episode_num == 0:  # prepare for save replay of evaluation
            self.env.close()
        #obs, pro_action, return, log_probs, values, advantage
        o, u, r, log_probs, values, advantage, padded, terminate, returns = [], [], [], [], [], [], [], [], []
        obs = self.env.reset()
        obs_ = (obs, self.env.game._flip_coord_observation_perspective(obs))  # A perspective, B perspective
        terminated = False
        win_tag = False
        step = 0
        episode_reward = (0, 0)  # cumulative rewards
        # last_action = np.zeros((self.args.n_agents, self.args.n_actions))
        # self.agents_sc.policy_base.init_hidden(1)

        # epsilon
        epsilon = 0 if evaluate else self.epsilon
        if self.args.epsilon_anneal_scale == 'episode':
            epsilon = epsilon - self.anneal_epsilon if epsilon > self.min_epsilon else epsilon
        actions_dist, log_probs, values, masks, rewards = [], [], [], [], []
        entropy = [0] * self.n_agents
        while not terminated and step < self.episode_limit:
            # time.sleep(0.2)
            obs = torch.tensor(obs_)    #A perspective, B perspective
            avail_action = [1] * self.args.n_actions
            action = [0] * self.args.n_agents
            action_dist, log_prob = [], []
            value = [0] * self.args.n_agents
            next_value = [0] * self.args.n_agents
            inputs = obs        #.copy()
            # transform the shape of inputs from (42,) to (1,42)
            for agent_id in range(self.n_agents):
                input = torch.tensor(inputs[agent_id], dtype=torch.float32).unsqueeze(0)
                if self.args.cuda:
                    input = input.cuda('cuda:3')
                # avail_actions = torch.tensor(avail_actions, dtype=torch.float32).unsqueeze(0)

                dist, value[agent_id] = self.ppo[agent_id](input)
                functional.reset_net(self.ppo[agent_id])

                action[agent_id] = torch.max(dist.sample(), 1)[1].cpu().numpy()[0]

                action_dist.append(dist.sample())  # tensor

                log_prob.append(dist.log_prob(dist.sample()).squeeze())
                entropy[agent_id] += dist.entropy().mean()

            obs_, reward, done, info = self.env.step(actions=action)
            terminate.append([terminated])

            action_dist = torch.stack(action_dist)  #[num_agnet, 1, num_actions(5]
            value = torch.stack(value).squeeze(1)   #[num_agnet, 1, num_values(1)]
            log_prob = torch.stack(log_prob)        #[num_agnet, 1, num_actions(5]

            o.append(obs)
            actions_dist.append(action_dist)
            log_probs.append(log_prob)
            values.append(value)
            masks.append(torch.FloatTensor(1 - np.array([done])).unsqueeze(1))
            rewards.append(torch.FloatTensor(reward).unsqueeze(1))

            episode_reward = [episode_reward[i] + reward[i] for i in range(min(len(episode_reward), len(reward)))]

            step += 1
            if self.args.epsilon_anneal_scale == 'step':
                epsilon = epsilon - self.anneal_epsilon if epsilon > self.min_epsilon else epsilon

        next_states = torch.tensor(obs_)
        for agent_id in range(self.n_agents):
            next_state = torch.tensor(next_states[agent_id], dtype=torch.float32).unsqueeze(0)
            if self.args.cuda:
                next_state = next_state.cuda('cuda:3')
            _, next_value[agent_id] = self.ppo[agent_id](next_state)
            functional.reset_net(self.ppo[agent_id])
        next_value = torch.stack(next_value)
        return_ = self.compute_gae(next_value, rewards, masks, values)
        return_ = torch.stack(return_)        #[num_steps, ]
        returns.append(return_)

        data = dict(o=o.copy(),                         #numpy.array
                    actions_dist=actions_dist.copy(),   #tensor
                    log_probs=log_probs.copy(),         #tensor
                    values=values.copy(),               #tensor
                    masks=masks.copy(),                 #tensor
                    rewards=rewards.copy(),             #tensor
                    # returns=returns.copy(),              #tensor
                    # terminated=terminate.copy()
                       )

        # add episode dim
        # for key in data.keys():
        #     data[key] = data[key]#np.array([])
        if not evaluate:
            self.epsilon = epsilon
        if evaluate and episode_num == self.args.evaluate_epoch - 1 and self.args.replay_dir != '':
            self.env.save_replay()
            self.env.close()

        return data, episode_reward, returns, step




    def compute_gae(self, next_value, rewards, masks, values, gamma=0.99, tau=0.95):
        values = values + [next_value.squeeze(1)]
        gae = 0
        return_ = []
        for step in reversed(range(len(rewards))):
            delta = rewards[step] + gamma * values[step + 1] * masks[step] - values[step]
            gae = delta + gamma * tau * masks[step] * gae
            return_.insert(0, gae + values[step])
        return return_