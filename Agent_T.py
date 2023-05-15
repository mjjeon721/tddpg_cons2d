import numpy as np

from model import *
from utils import *
import torch.optim as optim


class TDDPGAgent:
    def __init__(self, state_dim, action_dim, actor_lr=1e-6, critic_lr=0.001, \
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

    def update(self):
        # self.actor_optim.lr = 0.001 * 1 / (1 + 0.1 * (update_count // 1000))
        # self.critic_optim.param_groups[0]['lr'] = 1e-3 * 1 / (1 + 0.1 * (epoch // 1000))
        states, actions, rewards, utilities = self.history.sample(100)
        #actions = self.history.history['action']
        #U = self.history.history['utility'].reshape(-1)
        utilities = utilities.reshape(-1)
        rewards = rewards.reshape(-1)
        dU = np.zeros((actions.shape))
        # Policy updates
        # Computing dU
        for i in range(2):
            sorted_ix = np.argsort(self.history.history['action'][:, i])
            U_i = self.history.history['reward'][sorted_ix]
            actions_i = self.history.history['action'][sorted_ix, i]
            for j in range(len(dU)):
                val1, idx1 = find_nearest(actions_i, actions[j,i])
                U1 = U_i[idx1]
                while val1 == actions[j,i]:
                    actions_i = np.delete(actions_i, idx1)
                    U_i = np.delete(U_i, idx1)
                    val1, idx1 = find_nearest(actions_i, actions[j,i])
                    U1 = U_i[idx1]
                dU[j,i] = (U1 - rewards[j]) / (val1 - actions[j,i])
            # 1,i2 = np.argsort(np.abs(np.unique(actions[:,i]) - current_action[i]))[[0,1]]
            # tmp = (np.unique(U)[i1] - np.unique(U)[i2]) /( np.unique(actions[i1,i]) - np.unique(actions[i2, i]))
        #print(dU)
        dU = dU.reshape(1,-1)
        J_plus, J_minus = self.actor.policy_grad(states)
        d_plus_grad = np.sum(np.matmul(dU, J_plus).reshape(-1, 2), axis=0) / 100
        d_minus_grad = np.sum(np.matmul(dU, J_minus).reshape(-1, 2), axis=0) / 100
        # self.actor.d_plus = np.clip(self.actor_optim.update(update_count, self.actor.d_plus, d_plus_grad), 0, self.d_max)
        # self.actor.d_minus = np.clip(self.actor_optim.update(update_count, self.actor.d_minus, d_minus_grad), self.actor.d_plus, self.d_max)
        # self.actor.d_plus = np.clip(self.actor.d_plus + self.actor_lr * d_plus_grad, 0, self.d_max)
        self.actor.d_plus = self.actor.d_plus + self.actor_lr * d_plus_grad
        # self.actor.d_minus = np.clip(self.actor.d_minus + self.actor_lr * d_minus_grad, self.actor.d_plus, self.d_max)
        self.actor.d_minus = self.actor.d_minus + self.actor_lr * d_minus_grad
