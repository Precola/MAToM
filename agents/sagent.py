import numpy as np
import torch
from torch.distributions import Categorical
from common_sr.misc import gumbel_softmax
from braincog.base.encoder.population_coding import PEncoder
from spikingjelly.activation_based import functional
TIMESTEPS = 15
M = 5

# Agent
class Agents:
    def __init__(self, args):
        self.n_actions = args.n_actions
        self.n_agents = args.n_agents
        self.obs_shape = args.obs_shape
        # encoder
        self.pencoder = PEncoder(TIMESTEPS, 'population_voltage')

        if args.alg == 'svdn':
            from policy.svdn import SVDN
            self.policy = SVDN(args)
        elif args.alg == 'svdn_self':
            from policy.svdn_self import SVDN_SELF
            self.policy = SVDN_SELF(args)
        elif args.alg == 'stomvdn':
            from policy.stomvdn import SToMVDN
            self.policy = SToMVDN(args)
        elif args.alg == 'siql':
            from policy.siql import SIQL
            self.policy = SIQL(args)



        self.args = args

    def choose_action(self, num_env, obs, last_action, agent_num, avail_actions, epsilon, last_action_other, maven_z=None, evaluate=False):
        inputs = obs.copy()
        avail_actions_ind = np.nonzero(avail_actions)[0]  # index of actions which can be choose
        inputs_tom = obs.copy()
        # transform agent_num to onehot vector
        agent_id = np.zeros((num_env, self.n_agents))
        agent_id[:, agent_num] = 1.

        if self.args.last_action:
            inputs = np.hstack((inputs, last_action))
            inputs_tom = np.hstack((inputs_tom, last_action_other))
        if self.args.reuse_network:
            inputs = np.hstack((inputs, agent_id))

        # transform the shape of inputs from (42,) to (1,42)
        inputs = torch.tensor(inputs, dtype=torch.float32).unsqueeze(0)  # torch.Size([1, 17])
        inputs_tom = torch.tensor(inputs_tom, dtype=torch.float32)
        # init hidden tensor
        if self.args.alg == 'siql_e' or self.args.alg == 'siql_e2':
            h1_mem = self.policy.eval_h1_mem[:, agent_num, :, :, :, :]
            h1_spike = self.policy.eval_h1_spike[:, agent_num, :, :, :, :]
            h2_mem = self.policy.eval_h2_mem[:, agent_num, :, :, :, :]
            h2_spike = self.policy.eval_h2_spike[:, agent_num, :, :, :, :]
            inputs_, _ = self.pencoder(inputs=inputs, num_popneurons=M, VTH=0.99)    ###########################################################
            inputs = torch.transpose(inputs_, 0, 3)
            inputs = inputs.squeeze().unsqueeze(0)
        elif self.args.alg == 'svdn_self':
            tom_h1_mem = self.policy.tom_h1_mem
            tom_h1_spike = self.policy.tom_h1_spike
            tom_h2_mem = self.policy.tom_h2_mem
            tom_h2_spike = self.policy.tom_h2_spike
            h1_mem = self.policy.eval_h1_mem[:, agent_num, :, :]    #
            h1_spike = self.policy.eval_h1_spike[:, agent_num, :, :]
            h2_mem = self.policy.eval_h2_mem[:, agent_num, :, :]
            h2_spike = self.policy.eval_h2_spike[:, agent_num, :, :]
        elif self.args.alg == 'vdn_rnn':
            tom_h1_mem = self.policy.tom_h1_mem
            tom_h1_spike = self.policy.tom_h1_spike
            tom_h2_mem = self.policy.tom_h2_mem
            tom_h2_spike = self.policy.tom_h2_spike
            hidden_state = self.policy.eval_hidden[:, agent_num, :, :]
        else:
            h1_mem = self.policy.eval_h1_mem[:, agent_num, :, :]    #
            h1_spike = self.policy.eval_h1_spike[:, agent_num, :, :]
            h2_mem = self.policy.eval_h2_mem[:, agent_num, :, :]
            h2_spike = self.policy.eval_h2_spike[:, agent_num, :, :]

        avail_actions = torch.tensor(avail_actions, dtype=torch.float32).unsqueeze(0)
        if self.args.cuda:
            inputs_tom = inputs_tom.cuda(self.args.device)
            inputs = inputs.cuda(self.args.device)
            if self.args.alg == 'vdn_rnn' or self.args.alg == 'svdn_self':
                tom_h1_mem = tom_h1_mem.cuda(self.args.device)
                tom_h1_spike = tom_h1_spike.cuda(self.args.device)
                tom_h2_mem = tom_h2_mem.cuda(self.args.device)
                tom_h2_spike = tom_h2_spike.cuda(self.args.device)
            if self.args.alg != 'vdn_rnn' :
                h1_mem = h1_mem.cuda(self.args.device)
                h1_spike = h1_spike.cuda(self.args.device)
                h2_mem = h2_mem.cuda(self.args.device)
                h2_spike = h2_spike.cuda(self.args.device)
            else:
                hidden_state = hidden_state.cuda(self.args.device)


        if self.args.alg == 'svdn_self' or self.args.alg == 'vdn_rnn':
            a = inputs_tom.detach().clone()
            b = torch.cat((a[:, 2].unsqueeze(1), a[:, 3].unsqueeze(1), a[:, 0].unsqueeze(1), a[:, 1].unsqueeze(1), a[:, 4:]), 1)
            q_tom, self.policy.tom_h1_mem, self.policy.tom_h1_spike, self.policy.tom_h2_mem, self.policy.tom_h2_spike =\
                self.policy.tom_snn(b, tom_h1_mem, tom_h1_spike, tom_h2_mem, tom_h2_spike)
            action_tom = gumbel_softmax(q_tom, hard=True).unsqueeze(0)
            action_tom_ = torch.nn.Softmax(dim=1)(q_tom).unsqueeze(0)
            inputs = torch.cat((inputs, action_tom_), 2)

        # get q value
        if self.args.alg == 'siql_no_rnn' or self.args.alg == 'siql_no_rnn2':
            self.policy.eval_snn.reset()
            q_value,  = self.policy.eval_snn(inputs)
            # functional.reset_net(self.policy_sc.eval_snn)
        elif self.args.alg == 'vdn_rnn':
            q_value, self.policy.eval_hidden[:, agent_num, :] = self.policy.eval_rnn(inputs, hidden_state)
            q_value = q_value.unsqueeze(0)
        else:
            q_value, self.policy.eval_h1_mem[:, agent_num, :], self.policy.eval_h1_spike[:, agent_num, :],\
                self.policy.eval_h2_mem[:, agent_num, :], self.policy.eval_h2_spike[:, agent_num, :]= \
                self.policy.eval_snn(inputs, h1_mem, h1_spike, h2_mem, h2_spike)


        if np.random.uniform() < epsilon:
            # action = np.random.choice(avail_actions_ind)
            action = torch.tensor([[np.random.choice(avail_actions_ind) for i in range(num_env)]])
        else:
            action = torch.argmax(q_value, 2)
        return action


    def _choose_action_from_softmax(self, inputs, avail_actions, epsilon, evaluate=False):
        """
        :param_sc inputs: # q_value of all actions
        """
        action_num = avail_actions.sum(dim=1, keepdim=True).float().repeat(1, avail_actions.shape[-1])  # num of avail_actions
        prob = torch.nn.functional.softmax(inputs, dim=-1)
        # add noise of epsilon
        prob = ((1 - epsilon) * prob + torch.ones_like(prob) * epsilon / action_num)
        prob[avail_actions == 0] = 0.0


        if epsilon == 0 and evaluate:
            action = torch.argmax(prob)
        else:
            action = Categorical(prob).sample().long()
        return action

    def _get_max_episode_len(self, batch):
        terminated = batch['TERMINATE']
        episode_num = terminated.shape[0]
        max_episode_len = 0
        for episode_idx in range(episode_num):
            for transition_idx in range(self.args.episode_limit):
                if terminated[episode_idx, transition_idx, 0] == 1:
                    if transition_idx + 1 >= max_episode_len:
                        max_episode_len = transition_idx + 1
                    break
        if max_episode_len == 0:
            max_episode_len = self.args.episode_limit
        return max_episode_len

    def train(self, batch, train_step, epsilon=None):  # coma needs epsilon for training

        # different episode has different length, so we need to get max length of the batch
        max_episode_len = self._get_max_episode_len(batch)
        for key in batch.keys():
            if key != 'z':
                batch[key] = batch[key][:, :max_episode_len]
        self.policy.learn(batch, max_episode_len, train_step, epsilon)
        if train_step > 0 and train_step % self.args.save_cycle == 0:
            self.policy.save_model(train_step)



