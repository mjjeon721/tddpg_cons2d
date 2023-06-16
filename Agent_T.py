import numpy as np

from model import *
from utils import *
import torch.optim as optim
from scipy.spatial import KDTree


class TDDPGAgent:
    def __init__(self, state_dim, action_dim, actor_lr=1e-3, critic_lr=0.001, \
                 eta=1, d_max=1, tau=0.001, max_memory_size=10000):
        # Parameters
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.actor_lr = actor_lr

        self.eta = eta
        self.tau = tau

        self.d_max = d_max

        # Network object
        self.actor = Actor(self.state_dim, self.action_dim, self.d_max)
        self.critic = Critic(self.state_dim, self.action_dim)
        self.critic_target = Critic(self.state_dim, self.action_dim)

        # Target network initialization
        hard_updates(self.critic_target, self.critic)

        # Relay Buffer
        self.history = History()

        # Training models
        self.critic_criterion = nn.MSELoss()
        self.critic_optim = optim.Adam(self.critic.parameters(), critic_lr)

        self.actor_optim = AdamOptim(self.actor_lr)

    def get_action(self, state):
        state = Variable(torch.from_numpy(state).float())
        action = torch.squeeze(self.actor.forward(state))
        return action.detach().numpy()

    def random_action(self):
        return self.d_max * np.random.rand(self.action_dim)

    def update(self, current_state, current_action, current_reward, current_utility, update_count):
        self.actor_lr = 1e-3 * 1 / (1 + 0.1 * (update_count // 10000))
        # self.actor_optim.lr = 0.001 * 1 / (1 + 0.1 * (update_count // 1000))
        # self.critic_optim.param_groups[0]['lr'] = 1e-3 * 1 / (1 + 0.1 * (epoch // 1000))
        #states, actions, rewards, utilities = self.history.sample(100)
        #actions = self.history.history['action']
        #U = self.history.history['utility'].reshape(-1)
        #utilities = utilities.reshape(-1)
        #rewards = rewards.reshape(-1)

        # Policy updates
        # Computing dU
        dU = np.zeros((current_action.shape))
        actions = self.history.history['action']
        mask = np.all(actions != current_action, axis = 1)
        actions = actions[mask, :]
        utilities = self.history.history['utility'].reshape(-1)[mask]

        sorted_ix = np.argsort(actions[:,1])
        actions_i = actions[sorted_ix, 1]
        val, ix = find_nearest(actions_i, current_action[1])

        dU[0] = (current_utility - utilities[sorted_ix[ix]])/(current_action[0] - actions[sorted_ix[ix], 0])

        sorted_ix = np.argsort(actions[:, 0])
        actions_i = actions[sorted_ix, 0]
        val, ix = find_nearest(actions_i, current_action[0])
        dU[1] = (current_utility - utilities[sorted_ix[ix]]) / (current_action[1] - actions[sorted_ix[ix], 1])

        if current_state[0] < np.sum(current_action) :
            dr = dU - current_state[2]
        elif current_state[0] > np.sum(current_action):
            dr = dU - current_state[1]
        else :
            marginal_val = current_state[2] + (current_state[1] - current_state[2]) / (np.sum(self.actor.d_minus) - np.sum(self.actor.d_plus)) * (current_state[0] - np.sum(self.actor.d_plus))
            dr = dU - marginal_val

        J_plus, J_minus = self.actor.policy_grad(current_state)
        d_plus_grad = np.sum(np.matmul(dr, J_plus).reshape(-1, 2), axis=0)
        d_minus_grad = np.sum(np.matmul(dr, J_minus).reshape(-1, 2), axis=0)
        # self.actor.d_plus = np.clip(self.actor_optim.update(update_count, self.actor.d_plus, d_plus_grad), 0, self.d_max)
        # self.actor.d_minus = np.clip(self.actor_optim.update(update_count, self.actor.d_minus, d_minus_grad), self.actor.d_plus, self.d_max)
        # self.actor.d_plus = np.clip(self.actor.d_plus + self.actor_lr * d_plus_grad, 0, self.d_max)
        self.actor.d_plus = self.actor.d_plus + self.actor_lr * d_plus_grad
        # self.actor.d_minus = np.clip(self.actor.d_minus + self.actor_lr * d_minus_grad, self.actor.d_plus, self.d_max)
        self.actor.d_minus = self.actor.d_minus + self.actor_lr * d_minus_grad
