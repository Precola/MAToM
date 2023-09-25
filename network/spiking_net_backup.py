import torch
import torch.nn as nn
import torch.nn.functional as f
from torch.distributions import Normal

from spikingjelly.activation_based import neuron, functional
from spikingjelly.activation_based import rnn
from braincog.base.node.node import IFNode, LIFNode


thresh = 0.3
lens = 0.25
decay = 0.3
TIMESTEPS = 15
M = 5
# Neural Network
class NonSpikingLIFNode(neuron.LIFNode):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, dv: torch.Tensor):
        self.neuronal_charge(dv)
        # self.neuronal_fire()
        # self.neuronal_reset()
        return self.v

class BCNoSpikingLIFNode(LIFNode):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, dv: torch.Tensor):
        self.integral(dv)
        # self.neuronal_fire()
        # self.neuronal_reset()
        return self.mem

class ActorCritic(nn.Module):
    def __init__(self, num_inputs, args, T=16, std=0.0):   #num_outputs, hidden_size, T=16, std=0.0
        super(ActorCritic, self).__init__()

        self.critic = nn.Sequential(
            nn.Linear(num_inputs, args.ppo_hidden_size),
            neuron.IFNode(),
            nn.Linear(args.ppo_hidden_size, 1),
            NonSpikingLIFNode(tau=2.0)
        )

        self.actor = nn.Sequential(
            nn.Linear(num_inputs, args.ppo_hidden_size),
            neuron.IFNode(),
            nn.Linear(args.ppo_hidden_size, args.n_actions),
            NonSpikingLIFNode(tau=2.0)
        )

        self.log_std = nn.Parameter(torch.ones(1, args.n_actions) * std)

        self.T = T

    def forward(self, x):
        for t in range(self.T):
            self.critic(x)
            self.actor(x)
        value = self.critic[-1].v
        mu = self.actor[-1].v
        std = self.log_std.exp().expand_as(mu)
        dist = Normal(mu, std)
        return dist, value


class ActFun(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return input.gt(thresh).float()

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        temp = abs(input - thresh) < lens
        return grad_input * temp.float() / (2 * lens)


act_fun = ActFun.apply


def mem_update(fc, x, mem, spike):
    mem = mem * decay * (1 - spike) + fc(x)
    spike = act_fun(mem)
    return mem, spike

# class ActorCritic_rnn(nn.Module):
#     def __init__(self, num_inputs, args, T=16, std=0.0):   #num_outputs, hidden_size, T=16, std=0.0
#         super(ActorCritic_rnn, self).__init__()
#
#         self.h1_mem_actor = self.h1_spike_actor = self.h1_sumspike_actor = self.h1_summem_actor = torch.zeros(batch_size, args.ppo_hidden_size)
#         self.h2_mem_actor = self.h2_spike_actor = self.h2_sumspike_actor = self.h2_summem_actor = torch.zeros(batch_size, args.n_actions)
#
#         self.fc1_actor = nn.Linear(num_inputs, args.ppo_hidden_size, bias = True)
#         self.fc2_actor = nn.Linear(args.ppo_hidden_size, args.n_actions, bias = True)
#         # self.lateral1 = nn.Linear(cfg_fc[0], cfg_fc[0], bias = False)
#
#         self.log_std = nn.Parameter(torch.ones(1, args.n_actions) * std)
#
#         self.T = T
#
#         self.h1_mem_crtic = self.h1_spike_crtic = self.h1_sumspike_crtic = torch.zeros(batch_size, args.ppo_hidden_size)
#         self.h2_mem_crtic = self.h2_spike_crtic = self.h2_sumspike_crtic = torch.zeros(batch_size, 1)
#
#         self.fc1_crtic = nn.Linear(num_inputs, args.ppo_hidden_size, bias = True)
#         self.fc2_crtic = nn.Linear(args.ppo_hidden_size, 1, bias = True)
#         # self.lateral1 = nn.Linear(cfg_fc[0], cfg_fc[0], bias = False)

    # def forward(self, x):
    #
    #     x = x.view(x[0], -1)
    #
    #     self.h1_mem_actor, self.h1_spike_actor = mem_update(self.fc1_actor, x, self.h1_mem_actor, self.h1_spike_actor)
    #     self.h1_sumspike_actor = self.h1_sumspike_actor + self.h1_spike_actor
    #     self.h1_summem_actor = self.h1_summem_actor + self.h1_mem_actor
    #
    #     self.h2_mem_actor, self.h2_spike_actor = mem_update(self.fc2_actor, self.h1_spike_actor, self.h2_mem_actor, self.h2_spike_actor)
    #     self.h2_sumspike_actor = self.h2_sumspike_actor + self.h2_spike_actor
    #     self.h2_summem_actor = self.h2_summem_actor + self.h2_mem_actor
    #
    #     mu = self.h2_summem_actor
    #     std = self.log_std.exp().expand_as(mu)
    #     dist = Normal(mu, std)
    #
    #     self.h1_mem_critic, self.h1_spike_critic = mem_update(self.fc1_critic, x, self.h1_mem_critic, self.h1_spike_critic)
    #     self.h1_sumspike_critic = self.h1_sumspike_critic + self.h1_spike_critic
    #     self.h1_summem_critic = self.h1_summem_critic + self.h1_mem_critic
    #
    #     self.h2_mem_critic, self.h2_spike_critic = mem_update(self.fc2_critic, self.h1_spike_critic, self.h2_mem_critic, self.h2_spike_critic)
    #     self.h2_sumspike_critic = self.h2_sumspike_critic + self.h2_spike_critic
    #     self.h2_summem_critic = self.h2_summem_critic + self.h2_mem_critic
    #
    #     value = self.h2_summem_critic
    #
    #
    #     return self.h1_mem_actor, self.h1_spike_actor, self.h1_mem_critic, self.h1_spike_critic, dist, value
    #
    #
    #
    #
    #     value_x, _ = self.critic_lstm(x)
    #     actor_x, _ = self.actor_lstm(x)
    #     value      = self.critic_Node(self.critic_fc(value_x[-1]))[-1].v
    #     mu         = self.actor_Node (self.actor_fc(actor_x[-1]))[-1].v
    #
    #     std = self.log_std.exp().expand_as(mu)
    #     dist = Normal(mu, std)
    #     return dist, value

class Critic(nn.Module):
    def __init__(self, input_shape, args):
        super(Critic, self).__init__()
        self.args = args
        self.fc1 = nn.Linear(input_shape, args.ppo_hidden_size)#neuron.IFNode()
        self.fc2 = nn.Linear(args.ppo_hidden_size, args.ppo_hidden_size, bias = True)
        self.fc3 = nn.Linear(args.ppo_hidden_size, args.ppo_hidden_size, bias = True)
        self.fc4 = nn.Linear(args.ppo_hidden_size, args.n_actions)#
        self.req_grad = False

    def forward(self, inputs, h1_mem, h1_spike, h2_mem, h2_spike):
        # if self.req_grad == False:
        # [1, 17] -> [1, process, 64]
        x = self.fc1(inputs)
        # x = neuron.IFNode()(x)
        x = IFNode()(x)
        if self.args.alg == 'siql_e':
            h1_mem = h1_mem.reshape(-1,  M,TIMESTEPS, self.args.rnn_hidden_dim)
            h1_spike = h1_spike.reshape(-1,  M,TIMESTEPS, self.args.rnn_hidden_dim)
            h2_mem = h2_mem.reshape(-1, M, TIMESTEPS, self.args.rnn_hidden_dim)
            h2_spike = h2_spike.reshape(-1, M, TIMESTEPS, self.args.rnn_hidden_dim)

        else:
        # [1, 64] -> [process, 64]
            h1_mem = h1_mem.reshape(-1, self.args.rnn_hidden_dim)
            h1_spike = h1_spike.reshape(-1, self.args.rnn_hidden_dim)
            h2_mem = h2_mem.reshape(-1, self.args.rnn_hidden_dim)
            h2_spike = h2_spike.reshape(-1, self.args.rnn_hidden_dim)

        h1_mem, h1_spike = mem_update(self.fc2, x, h1_mem, h1_spike)
        h2_mem, h2_spike = mem_update(self.fc3, h1_spike, h2_mem, h2_spike)
        # [1, 5]
        # value = NonSpikingLIFNode(tau=2.0)(self.fc4(h2_mem))
        value = BCNoSpikingLIFNode(tau=2.0)(self.fc4(h2_mem))

        return value, h1_mem, h1_spike, h2_mem, h2_spike

class Critic2(nn.Module):
    def __init__(self, input_shape, args):
        super(Critic2, self).__init__()
        self.args = args
        # self.fc1 = nn.Linear(input_shape, args.ppo_hidden_size)#neuron.IFNode()
        self.fc2 = nn.Linear(input_shape, args.ppo_hidden_size, bias = True)
        self.fc3 = nn.Linear(args.ppo_hidden_size, args.ppo_hidden_size, bias = True)
        self.fc4 = nn.Linear(args.ppo_hidden_size, args.n_actions)#
        self.req_grad = False

    def forward(self, inputs, h1_mem, h1_spike, h2_mem, h2_spike):
        if self.req_grad == False:
            # [1, 17] -> [1, process, 64]
            # x = self.fc1(inputs)
            # x = neuron.IFNode()(x)
            x = inputs
            if self.args.alg == 'siql_e' or self.args.alg == 'siql_e2':
                h1_mem = h1_mem.reshape(-1,  M,TIMESTEPS, self.args.rnn_hidden_dim)
                h1_spike = h1_spike.reshape(-1,  M,TIMESTEPS, self.args.rnn_hidden_dim)
                h2_mem = h2_mem.reshape(-1, M, TIMESTEPS, self.args.rnn_hidden_dim)
                h2_spike = h2_spike.reshape(-1, M, TIMESTEPS, self.args.rnn_hidden_dim)

            else:
            # [1, 64] -> [process, 64]
                h1_mem = h1_mem.reshape(-1, self.args.rnn_hidden_dim)
                h1_spike = h1_spike.reshape(-1, self.args.rnn_hidden_dim)
                h2_mem = h2_mem.reshape(-1, self.args.rnn_hidden_dim)
                h2_spike = h2_spike.reshape(-1, self.args.rnn_hidden_dim)

            h1_mem, h1_spike = mem_update(self.fc2, x, h1_mem, h1_spike)
            h2_mem, h2_spike = mem_update(self.fc3, h1_spike, h2_mem, h2_spike)
            # [1, 5]
            value = NonSpikingLIFNode(tau=2.0)(self.fc4(h2_mem))

        else:
            with torch.no_grad():
                # [1, 17]
                x = self.fc1(inputs)
                x = neuron.IFNode()(x)
                # [1, 64]
                h1_mem = h1_mem.reshape(-1, self.args.rnn_hidden_dim)
                h1_spike = h1_spike.reshape(-1, self.args.rnn_hidden_dim)
                h2_mem = h2_mem.reshape(-1, self.args.rnn_hidden_dim)
                h2_spike = h2_spike.reshape(-1, self.args.rnn_hidden_dim)
                h1_mem, h1_spike = mem_update(self.fc2, x, h1_mem, h1_spike)

                h2_mem, h2_spike = mem_update(self.fc3, h1_spike, h2_mem, h2_spike)
                # [1, 5]
                value = NonSpikingLIFNode(tau=2.0)(self.fc4(h2_mem))

        return value, h1_mem, h1_spike, h2_mem, h2_spike

class Critic_without_recurrent(nn.Module):
    def __init__(self, input_shape, args, T=16):
        super(Critic_without_recurrent, self).__init__()
        self._node = IFNode
        self.v_reset = 0.
        self.args = args
        self.fc = nn.Sequential(
            nn.Linear(input_shape, args.ppo_hidden_size),
            self._node(v_reset=self.v_reset),
            nn.Linear(args.ppo_hidden_size, args.ppo_hidden_size),
            self._node(v_reset=self.v_reset),
            nn.Linear(args.ppo_hidden_size, args.ppo_hidden_size),
            self._node(v_reset=self.v_reset),
            nn.Linear(args.ppo_hidden_size, args.n_actions),
            BCNoSpikingLIFNode(tau=2.0)
        )

        self.T = T

    def reset(self):
        for mod in self.modules():
            if hasattr(mod, 'n_reset'):
                mod.n_reset()

    def forward(self, x):
        if self.args.alg == 'siql_no_rnn2':
            self.reset()

        for t in range(self.T):
            self.fc(x)

        return self.fc[-1].mem

class VDNNet(nn.Module):
    def __init__(self):
        super(VDNNet, self).__init__()

    def forward(self, q_values):
        return torch.sum(q_values, dim=2, keepdim=True)