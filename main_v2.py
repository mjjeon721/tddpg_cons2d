import torch
import torch.autograd
import torch.optim as optim
import torch.nn as nn
import numpy as np
import sys
from scipy import io
import matplotlib.pyplot as plt
from scipy.stats import truncnorm
from collections import OrderedDict
from model import *
from DDPG_model import *
from utils import *
import time
import pickle
np.random.seed(0)

# Random prices but finite number of possible prices (Randomly sample from 100 possible prices)

class DDPGAgent :
    def __init__(self, state_dim, action_dim, actor_lr= 1e-4, critic_lr=1e-3, \
                 eta=1, d_max = 1, tau=0.001, max_memory_size=50000):
        # Parameters
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.actor_lr = actor_lr

        self.eta = eta
        self.tau = tau

        self.d_max = d_max

        # Network object
        self.actor = Actor_d(self.state_dim, self.action_dim, self.d_max)
        self.actor_target = Actor_d(self.state_dim, self.action_dim, self.d_max)
        self.critic = Critic_d(self.state_dim, self.action_dim)
        self.critic_target = Critic_d(self.state_dim, self.action_dim)

        # Target network initialization
        hard_updates(self.critic_target, self.critic)
        hard_updates(self.actor_target, self.actor)

        # Training
        self.memory = Memory(max_memory_size)
        self.critic_criterion = nn.MSELoss()
        self.actor_optim = optim.Adam(self.actor.parameters(), actor_lr)
        self.critic_optim = optim.Adam(self.critic.parameters(), critic_lr)

    def get_action(self, state):
        state = Variable(torch.from_numpy(state).float())
        action = self.actor.forward(state)
        return action.detach().numpy()

    def random_action(self):
        return self.d_max * np.random.rand(2)

    def update(self, batch_size, epoch):
        #self.actor_optim.param_groups[0]['lr'] = 1e-4 * 1 / (1 + 0.1 * (epoch // 1000))
        #self.critic_optim.param_groups[0]['lr'] = 1e-3 * 1 / (1 + 0.1 * (epoch // 1000))
        states, actions, rewards, next_states, dones = self.memory.sample(batch_size)
        states = torch.FloatTensor(np.vstack(states))
        actions = torch.FloatTensor(np.vstack(actions))
        rewards = torch.FloatTensor(np.vstack(rewards))
        next_states = torch.FloatTensor(np.vstack(next_states))
        dones = torch.Tensor(np.array(dones)).view(-1,1)

        # Critic loss
        Qvals = self.critic.forward(states, actions)
        next_actions = self.actor_target.forward(next_states)
        next_Q = self.critic_target.forward(next_states, next_actions)
        Qprime = rewards + self.eta * next_Q
        critic_loss = self.critic_criterion(Qvals, Qprime)

        # Critic updates
        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()

        policy_loss = - self.critic.forward(states, self.actor.forward(states)).mean()

        self.actor_optim.zero_grad()
        policy_loss.backward()
        self.actor_optim.step()

        soft_updates(self.critic_target, self.critic, self.tau)
        soft_updates(self.actor_target, self.actor, self.tau)

class TDDPGAgent:
    def __init__(self, state_dim, action_dim, actor_lr = 1e-4, critic_lr=1e-3, \
                 eta=1, d_max = 1, tau=0.001, max_memory_size = 50000):
        # Parameters
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.actor_lr = actor_lr

        self.eta = eta
        self.tau = tau

        self.d_max = d_max

        # Network object
        self.actor = Actor(self.state_dim, self.action_dim, self.d_max)
        self.actor_target = Actor(self.state_dim, self.action_dim, self.d_max)
        self.critic = Critic(self.state_dim, self.action_dim)
        self.critic_target = Critic(self.state_dim, self.action_dim)

        # Target network initialization
        hard_updates(self.critic_target, self.critic)
        hard_updates(self.actor_target, self.actor)

        # Relay Buffer
        self.memory = Memory(max_memory_size)

        # Training models
        self.critic_criterion = nn.MSELoss()
        self.actor_optim = optim.Adam(self.actor.parameters(), actor_lr)
        self.critic_optim = optim.Adam(self.critic.parameters(), critic_lr)

    def get_action(self, state):
        state = Variable(torch.from_numpy(state).float())
        action = torch.squeeze(self.actor.forward(state))
        return action.detach().numpy()

    def random_action(self):
        return self.d_max * np.random.rand(self.action_dim)

    def update(self, batch_size, epoch):
        #self.actor_optim.param_groups[0]['lr'] = 1e-4 * 1 / (1 + 0.1 * (epoch // 1000))
        #self.critic_optim.param_groups[0]['lr'] = 1e-3 * 1 / (1 + 0.1 * (epoch // 1000))
        states, actions, rewards, next_states, dones = self.memory.sample(batch_size)
        states = torch.FloatTensor(np.vstack(states))
        actions = torch.FloatTensor(np.vstack(actions))
        rewards = torch.FloatTensor(np.vstack(rewards))
        next_states = torch.FloatTensor(np.vstack(next_states))
        dones = torch.Tensor(np.array(dones)).view(-1,1)

        Qvals = self.critic.forward(states, actions)
        next_actions = self.actor_target.forward(next_states)

        next_Q = self.critic_target.forward(next_states, next_actions)
        Qprime = rewards + self.eta * next_Q
        critic_loss = self.critic_criterion(Qvals, Qprime)

        # Critic updates
        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()

        # Policy updates
        policy_loss = -self.critic.forward(states, self.actor.forward(states)).mean()
        self.actor_optim.zero_grad()
        policy_loss.backward()
        self.actor_optim.step()

        soft_updates(self.critic_target, self.critic, self.tau)
        soft_updates(self.actor_target, self.actor, self.tau)
class Env:
    def __init__(self, util_param, renewable_param, reward_max):
        self.a = util_param[0]
        self.b = util_param[1]
        self.renewable_param = np.array(renewable_param)
        self.reward_max = reward_max

    def get_next_state(self):
        #r_next = np.random.uniform(0, 3, 1)#truncnorm.rvs(-self.renewable_param[0] / self.renewable_param[1], 100, size=1) * self.renewable_param[1] + self.renewable_param[0]
        r_next = truncnorm.rvs(-self.renewable_param[0] / self.renewable_param[1], (self.renewable_param[2] - self.renewable_param[0]) / self.renewable_param[1], size=1) * self.renewable_param[1] + self.renewable_param[0]
        return r_next

    def get_reward(self, state, action):
        net_consum = np.sum(action) - state[0]
        reward = np.matmul(self.a, action) - 0.5 * np.matmul(self.b , action ** 2) - np.max(np.array((state[2], state[1])) * net_consum)
        return reward / self.reward_max

# x_t = r_t, pi_t
# a_t = d_t (2-D Consumption)
state_dim = 1 + 2
action_dim = 2

# Action constraint parameters
d_max = 3

agent_tddpg = TDDPGAgent(state_dim, action_dim)
agent_ddpg = DDPGAgent(state_dim, action_dim)

# Batch sample size
batch_size = 100

T = 10
# Renewable parameters
r_mean = 2 / d_max
r_std = 3 / d_max
r_max = 4.5 / d_max

# Number of interaction with environment
interaction = 0

# Model parameters
# NEM parameters

NEM_param = 0.5 * np.sort(np.random.rand(600)).reshape(2,-1)

a = np.array([3, 2.7]) / d_max
b = np.array([1, 1])

opt_action_1 = a[0] - NEM_param
opt_action_2 = a[1] - NEM_param

opt_d_plus = np.array([opt_action_1[1,:], opt_action_2[1,:]])
opt_d_minus = np.array([opt_action_1[0,:], opt_action_2[0,:]])
reward_max = max(np.sum(a.reshape(2,1) * opt_d_minus - 0.5 * opt_d_minus **2, axis = 0) - NEM_param[0,:] * (np.sum(opt_d_minus, axis = 0) - r_max))

env = Env([a,b], [r_mean, r_std, r_max], reward_max)

tic = time.perf_counter()

epoch_size = 1000
num_epoch = 200
update_count = 0

DDPG_reward = []
DDPG_avg_reward = []

TDDPG_reward = []
TDDPG_avg_reward = []

OPT_reward = []
OPT_avg_reward = []

for epoch in range(num_epoch) :
    epoch_reward_DDPG = 0
    epoch_reward_TDDPG = 0
    epoch_reward_OPT = 0
    r_0_samples_ddpg = truncnorm.rvs(-r_mean/ r_std, (r_max - r_mean)/r_std, size=epoch_size) * r_std + r_mean
    r_0_samples_opt = truncnorm.rvs(-r_mean / r_std, (r_max - r_mean) / r_std, size=epoch_size) * r_std + r_mean
    r_0_samples_tddpg = truncnorm.rvs(-r_mean / r_std, (r_max - r_mean) / r_std, size=epoch_size) * r_std + r_mean
    for episode in range(epoch_size):
        ix_ddpg = np.random.randint(NEM_param.shape[1])
        ix_tddpg = np.random.randint(NEM_param.shape[1])
        ix_opt = np.random.randint(NEM_param.shape[1])
        state_ddpg = np.array([r_0_samples_ddpg[episode], NEM_param[0,ix_ddpg], NEM_param[1,ix_ddpg]])
        state_tddpg = np.array([r_0_samples_tddpg[episode], NEM_param[0, ix_tddpg], NEM_param[1, ix_tddpg]])
        state_opt = np.array([r_0_samples_opt[episode], NEM_param[0, ix_opt], NEM_param[1, ix_opt]])

        if interaction <= 10000 :
            action_tddpg = agent_tddpg.random_action()
            action_ddpg = agent_ddpg.random_action()
        else:
            explr_noise_std = 0.1
            action_tddpg = np.clip(agent_tddpg.get_action(state_ddpg) + explr_noise_std * np.random.randn(2), 0, 1)
            action_ddpg = np.clip(agent_ddpg.get_action(state_tddpg) + explr_noise_std * np.random.randn(2), 0, 1)

        # Observing next state and rewards
        ix_n_ddpg = np.random.randint(NEM_param.shape[1])
        ix_n_tddpg = np.random.randint(NEM_param.shape[1])
        ix_n_opt = np.random.randint(NEM_param.shape[1])
        new_state_ddpg = np.append(env.get_next_state(), [NEM_param[0,ix_n_ddpg], NEM_param[1,ix_n_ddpg]])
        new_state_tddpg = np.append(env.get_next_state(), [NEM_param[0, ix_n_tddpg], NEM_param[1, ix_n_tddpg]])
        new_state_opt = np.append(env.get_next_state(), [NEM_param[0, ix_n_opt], NEM_param[1, ix_n_opt]])
        reward_tddpg = env.get_reward(state_tddpg, action_tddpg).reshape(1,)
        reward_ddpg = env.get_reward(state_ddpg, action_ddpg).reshape(1,)
        done = True if episode == epoch_size-1 else False
        if state_opt[0] < sum(opt_d_plus[:,ix_opt]) :
            action_opt = opt_d_plus[:,ix_opt]
        elif state_opt[0] > sum(opt_d_minus[:,ix_opt]) :
            action_opt = opt_d_minus[:,ix_opt]
        else :
            action_opt = opt_d_plus[:,ix_opt] + 0.5 * (state_opt[0] - sum(opt_d_plus[:,ix_opt]))
        reward_opt = env.get_reward(state_opt, action_opt).reshape(1,)

        epoch_reward_DDPG += reward_ddpg
        epoch_reward_TDDPG += reward_tddpg
        epoch_reward_OPT += reward_opt

        # Storing in replay buffer
        agent_tddpg.memory.push(state_tddpg, action_tddpg, reward_tddpg, new_state_tddpg, done)
        agent_ddpg.memory.push(state_ddpg, action_ddpg, reward_ddpg, new_state_ddpg, done)
        interaction+=1
        state_ddpg = new_state_ddpg
        state_tddpg = new_state_tddpg
        state_opt = new_state_opt
        ix_ddpg = ix_n_ddpg
        ix_tddpg = ix_n_tddpg
        ix_opt = ix_n_opt

        if interaction > 1000 and (interaction%20 == 1):
            for grad_update in range(20):
                agent_tddpg.update(batch_size, update_count)
                agent_ddpg.update(batch_size, update_count)
            update_count += 1
    DDPG_reward.append(epoch_reward_DDPG)
    DDPG_avg_reward.append(np.mean(DDPG_reward[-100:]))
    TDDPG_reward.append(epoch_reward_TDDPG)
    TDDPG_avg_reward.append(np.mean(TDDPG_reward[-100:]))
    OPT_reward.append(epoch_reward_OPT)
    OPT_avg_reward.append(np.mean(OPT_reward[-100:]))

    # End of epoch, policy evaluation phase
    # Number of episodes : 100
    # Number of seeds : 1
    if epoch % 20 == 19 :
        toc = time.perf_counter()
        print('1 Epoch running time : {0:.4f} (s)'.format(toc - tic))
        print('Epoch : {0}, TDDPG_avg_reward : {1:.4f}, DDPG_avg_reward : {2:.4f} Optimal_avg_reward : {3:.4f}'. format(epoch, TDDPG_avg_reward[-1], DDPG_avg_reward[-1], OPT_avg_reward[-1]))
        tic = time.perf_counter()
'''
smoothed_curve_ddpg = np.array([])
smoothed_curve_tddpg = np.array([])
smoothed_curve_opt = np.array([])
for i in range(num_epoch) :
    smoothed_curve_ddpg = np.append(smoothed_curve_ddpg, np.mean(DDPG_avg_reward[np.maximum(i-10, 0):i+1]))
    smoothed_curve_tddpg = np.append(smoothed_curve_tddpg, np.mean(TDDPG_avg_reward[np.maximum(i - 10, 0):i + 1]))
    smoothed_curve_opt = np.append(smoothed_curve_opt, np.mean(OPT_avg_reward[np.maximum(i - 10, 0):i + 1]))
plt.plot(np.arange(0, 200000, 1000),smoothed_curve_tddpg, label = 'TDDPG')
plt.plot(np.arange(0, 200000, 1000),smoothed_curve_ddpg, label = 'DDPG')
plt.plot(np.arange(0, 200000, 1000),smoothed_curve_opt, label = 'Optimal')
plt.legend()
plt.xlabel('Step')
plt.ylabel('Performance')
plt.grid()
plt.show()

regret_TDDPG = (smoothed_curve_opt - smoothed_curve_tddpg) / smoothed_curve_opt * 100
regret_DDPG = (smoothed_curve_opt - smoothed_curve_ddpg) / smoothed_curve_opt * 100
plt.plot(np.arange(0, 200000, 1000),regret_TDDPG, label = 'TDDPG')
plt.plot(np.arange(0, 200000, 1000),regret_DDPG, label = 'DDPG')
plt.legend()
plt.xlabel('Step')
plt.ylabel('Regret (%)')
#plt.ylim(bottom = 0, top = 2)
plt.grid()
plt.show()

plt.plot(np.arange(0, 250000, 1000),smoothed_curve_tddpg, label = 'TDDPG')
plt.plot(np.arange(0, 250000, 1000),smoothed_curve_ddpg, label = 'DDPG')
plt.plot(np.arange(0, 250000, 1000),smoothed_curve_opt, label = 'Optimal')
plt.legend()
plt.xlabel('Step')
plt.ylabel('Performance')
plt.grid()
plt.show()

regret_TDDPG = (smoothed_curve_opt - smoothed_curve_tddpg) / smoothed_curve_opt * 100
regret_DDPG = (smoothed_curve_opt - smoothed_curve_ddpg) / smoothed_curve_opt * 100
regret_TDDPG = (learning_curve_opt - learning_curve_tddpg) / learning_curve_opt * 100
regret_DDPG = (learning_curve_opt - learning_curve_ddpg) / learning_curve_opt * 100
plt.plot(np.arange(0, 250000, 1000),regret_TDDPG, label = 'TDDPG')
plt.plot(np.arange(0, 250000, 1000),regret_DDPG, label = 'DDPG')
plt.legend()
plt.xlabel('Step')
plt.ylabel('Regret (%)')
plt.ylim(bottom = 0, top = 2)
plt.grid()
plt.show()

torch.save(agent_ddpg.actor, 'ddpg_trained.pth')
torch.save(agent_tddpg.actor, 'tddpg_trained.pth')

f = open('learning_curve_ddpg.pckl','wb')
pickle.dump(learning_curve_ddpg,f)
f.close()

f = open('learning_curve_tddpg.pckl','wb')
pickle.dump(learning_curve_tddpg,f)
f.close()
f = open('learning_curve_opt.pckl','wb')
pickle.dump(learning_curve_opt,f)
f.close()

jx = np.random.randint(NEM_param.shape[1], size = 10)

x = torch.arange(0, 3, 0.01)
trained_output_TDDPG = []
trained_output_DDPG = []
for j in range(10) :
    jj = jx[j]
    for i in range(len(x)) :
        trained_output_TDDPG.append(agent_tddpg.actor.forward(torch.Tensor([x[i], NEM_param[0,jj], NEM_param[1,jj]])).detach().numpy())
        trained_output_DDPG.append(
            agent_ddpg.actor.forward(torch.Tensor([x[i], NEM_param[0, jj], NEM_param[1, jj]])).detach().numpy())

trained_output_TDDPG = np.vstack(trained_output_TDDPG)
trained_output_DDPG = np.vstack(trained_output_DDPG)
i = 8
plt.plot(x, np.sum(trained_output_TDDPG[i * x.shape[0]:(i+1) * x.shape[0],:], axis = 1) - x.numpy(), label = 'TDDPG')
plt.plot(x, np.sum(trained_output_DDPG[i * x.shape[0]:(i+1) * x.shape[0],:], axis = 1) - x.numpy(), label = 'DDPG')
plt.plot(x, np.maximum(np.minimum(x, sum(opt_d_minus[:,i])), sum(opt_d_plus[:,i])) - x.numpy(), label = 'OPT_policy')
plt.xlabel('Renewables')
plt.ylabel('Net Consumption')
plt.legend()
plt.grid()
plt.show()

plt.plot(x, trained_output_TDDPG[i * x.shape[0]:(i+1) * x.shape[0],1], label = 'TDDPG')
plt.plot(x, trained_output_DDPG[i * x.shape[0]:(i+1) * x.shape[0],1], label = 'DDPG')
plt.plot(x, np.maximum(np.minimum(x, opt_d_minus[1,i]), opt_d_plus[1,i]), label = 'Opt_policy' )
plt.xlabel('Renewables')
plt.ylabel('$d_2$')
plt.legend()
plt.grid()
plt.show()

plt.plot(x, trained_output_TDDPG[i * x.shape[0]:(i+1) * x.shape[0],0], label = 'TDDPG')
plt.plot(x, trained_output_DDPG[i * x.shape[0]:(i+1) * x.shape[0],0], label = 'DDPG')
plt.plot(x, np.maximum(np.minimum(x, opt_d_minus[0,i]), opt_d_plus[0,i]), label = 'Opt_policy' )
plt.xlabel('Renewables')
plt.ylabel('$d_1$')
plt.legend()
plt.grid()
plt.show()

# Thresholds plot
plt.plot(NEM_param[1,:], d_plus_learned[:,0], label = '$d_1^+ learned$')
plt.plot(NEM_param[1,:], opt_d_plus[0,:], label = '$d_1^*$')
plt.xlabel('$\pi_t^+$')
plt.legend()
plt.grid()
plt.show()

plt.plot(NEM_param[1,:], d_plus_learned[:,1], label = '$d_2^+ learned$')
plt.plot(NEM_param[1,:], opt_d_plus[1,:], label = '$d_2^*$')
plt.xlabel('$\pi_t^+$')
plt.legend()
plt.grid()
plt.show()

plt.plot(NEM_param[0,:],d_minus_learned[:,0], label = '$d_1^- learned$')
plt.plot(NEM_param[0,:],opt_d_minus[0,:], label = '$d_1^{-*}$')
plt.xlabel('$\pi_t^-$')
plt.grid()
plt.legend()
plt.show()

plt.plot(NEM_param[0,:],d_minus_learned[:,1], label = '$d_2^+ learned$')
plt.plot(NEM_param[0,:],opt_d_minus[1,:], label = '$d_2^{-*}$')
plt.xlabel('$\pi_t^-$')
plt.grid()
plt.legend()
plt.show()

plt.plot(np.arange(0, 250000, 1000),learning_curve_tddpg, label = 'TDDPG')
plt.plot(np.arange(0, 250000, 1000),learning_curve_ddpg, label = 'DDPG')
plt.plot(np.arange(0, 250000, 1000),learning_curve_opt, label = 'Optimal')
plt.legend()
plt.xlabel('Step')
plt.ylabel('Performance')
plt.grid()
plt.show()

regret_TDDPG = (learning_curve_opt - learning_curve_tddpg) / learning_curve_opt * 100
regret_DDPG = (learning_curve_opt - learning_curve_ddpg) / learning_curve_opt * 100
plt.plot(np.arange(0, 250000, 1000),regret_TDDPG, label = 'TDDPG')
plt.plot(np.arange(0, 250000, 1000),regret_DDPG, label = 'DDPG')
plt.legend()
plt.xlabel('Step')
plt.ylabel('Regret (%)')
#plt.ylim(bottom = 0, top = 2)
plt.grid()
plt.show()

for episode in range(max_epsiodes):
    episode_reward_tddpg = 0
    episode_reward_ddpg = 0
    opt_episode_reward = 0
    ix = np.random.randint(NEM_param.shape[1])
    state = np.array([r_0_samples[episode], NEM_param[0,ix], NEM_param[1,ix]])
    for step in range(T):
        if interaction <= 5000:
            action_tddpg = agent_tddpg.random_action()
            action_ddpg = agent_ddpg.random_action()

        else:
            explr_noise_std = 0.1 if episode <= 1000 else 1e-1 / np.sqrt(episode / 1000)
            action_tddpg = np.clip(agent_tddpg.get_action(state) + explr_noise_std * np.random.randn(2), 0, 1)
            action_ddpg = np.clip(agent_ddpg.get_action(state) + explr_noise_std * np.random.randn(2), 0, 1)

        # Observing next state and rewards
        ix_n = np.random.randint(NEM_param.shape[1])
        new_state = np.append(env.get_next_state(), [NEM_param[0,ix_n], NEM_param[1,ix_n]])
        reward_tddpg = env.get_reward(state, action_tddpg).reshape(1,)
        reward_ddpg = env.get_reward(state, action_ddpg).reshape(1,)
        done = True if step == T-1 else False
        agent_tddpg.memory.push(state, action_tddpg, reward_tddpg, new_state, done)
        agent_ddpg.memory.push(state, action_ddpg, reward_ddpg, new_state, done)
        interaction+=1

        if state[0] <= sum(opt_d_plus[:,ix]):
            opt_action = opt_d_plus[:,ix]
        elif state[0] >= sum(opt_d_minus[:,ix]):
            opt_action = opt_d_minus[:,ix]
        else:
            opt_action = opt_d_plus[:,ix] + 0.5 * (state[0] - sum(opt_d_plus[:,ix]))

        opt_episode_reward += env.get_reward(state, opt_action) * agent_tddpg.eta ** step

        state = new_state
        ix = ix_n
        episode_reward_tddpg += reward_tddpg.reshape(1,) * agent_tddpg.eta ** step
        episode_reward_ddpg += reward_ddpg.reshape(1,) * agent_ddpg.eta ** step

    if interaction > 6000 :
        for grad_update in range(10):
            agent_tddpg.update(batch_size)
            agent_ddpg.update(batch_size)

    rewards_tddpg.append(episode_reward_tddpg)
    rewards_ddpg.append(episode_reward_ddpg)
    opt_rewards.append(opt_episode_reward)
    avg_rewards_tddpg.append(np.mean(rewards_tddpg[-100:]))
    avg_rewards_ddpg.append(np.mean(rewards_ddpg[-100:]))
    opt_avg_rewards.append(np.mean(opt_rewards[-100:]))

    if episode % 500 == 499 :
        toc = time.perf_counter()
        print('500 Episodes running time : {0:.4f} (s)'.format(toc - tic))
        print('Episode : {0}, TDDPG_avg_reward : {1:.4f}, DDPG_avg_reward : {2:.4f} Optimal_avg_reward : {3:.4f}'. format(episode, avg_rewards_tddpg[-1], avg_rewards_ddpg[-1], opt_avg_rewards[-1]))
        tic = time.perf_counter()


plt.plot(avg_rewards_tddpg, label='TDDPG')
plt.plot(avg_rewards_ddpg, label = 'DDPG')
plt.plot(opt_avg_rewards, label='OPT_policy')
plt.xlabel('episodes')
plt.ylabel('rewards')
plt.legend()
plt.grid()
plt.show()

regret_TDDPG = np.abs(np.array(avg_rewards_tddpg) - np.array(opt_avg_rewards)) / np.array(opt_avg_rewards) * 100
regret_DDPG = np.abs(np.array(avg_rewards_ddpg) - np.array(opt_avg_rewards)) / np.array(opt_avg_rewards) * 100
plt.plot(regret_TDDPG,label = 'TDDPG')
plt.plot(regret_DDPG, label = 'DDPG')
plt.legend()
plt.ylabel('Regret (%)')
#plt.ylim(top = 5)
plt.grid()
plt.show()

threshold_result = []
# Threshold learned result
for i in range(NEM_param.shape[1]) :
    price = torch.FloatTensor(NEM_param[:,i])
    out = agent_tddpg.actor.fc1_th(price)
    out = agent_tddpg.actor.relu(out)
    out = agent_tddpg.actor.fc2_th(out)
    out = agent_tddpg.actor.relu(out)
    out = agent_tddpg.actor.fc3_th(out)
    out = agent_tddpg.actor.sigmoid(out)
    threshold_result.append(out.detach().numpy())

threshold_result = np.vstack(threshold_result)
d_plus_learned = threshold_result[:,[0,2]]
d_minus_learned = threshold_result[:,[1,3]]

# Figure plots
x = torch.arange(0, 3, 0.01)
trained_output_TDDPG = []
for j in range(10) :
    for i in range(len(x)) :
        trained_output_TDDPG.append(agent_tddpg.actor.forward(torch.Tensor([x[i], NEM_param[0,j], NEM_param[1,j]])).detach().numpy())

trained_output_TDDPG = np.vstack(trained_output_TDDPG)
i = 8
plt.plot(x, np.sum(trained_output_TDDPG[i * x.shape[0]:(i+1) * x.shape[0],:], axis = 1) - x.numpy(), label = 'TDDPG')
plt.plot(x, np.maximum(np.minimum(x, sum(opt_d_minus[:,i])), sum(opt_d_plus[:,i])) - x.numpy(), label = 'OPT_policy')
plt.xlabel('Renewables')
plt.ylabel('Net Consumption')
plt.legend()
plt.grid()
plt.show()

plt.plot(x, trained_output_TDDPG[i * x.shape[0]:(i+1) * x.shape[0],1], label = 'TDDPG')
plt.plot(x, np.maximum(np.minimum(x, opt_d_minus[1,i]), opt_d_plus[1,i]), label = 'Opt_policy' )
plt.xlabel('Renewables')
plt.ylabel('$d_2$')
plt.legend()
plt.grid()
plt.show()

plt.plot(x, trained_output_TDDPG[i * x.shape[0]:(i+1) * x.shape[0],0], label = 'TDDPG')
plt.plot(x, np.maximum(np.minimum(x, opt_d_minus[0,i]), opt_d_plus[0,i]), label = 'Opt_policy' )
plt.xlabel('Renewables')
plt.ylabel('$d_1$')
plt.legend()
plt.grid()
plt.show()



plt.plot(np.sum(d_minus_learned, axis = 1), label = '$d^- learned$')
plt.plot(np.sum(opt_d_minus, axis = 0),label =  '$d^{-*}$')
plt.legend()
plt.show()

plt.plot(np.sum(d_plus_learned, axis = 1), label = '$d^+ learned$')
plt.plot(np.sum(opt_d_plus, axis = 0),label =  '$d^{+*}$')
plt.legend()
plt.show()

'''