import numpy as np

from model import *
from utils import *
import torch.optim as optim


class TDDPGAgent:
    def __init__(self, state_dim, action_dim, actor_lr=1e-3, reward_lr=0.001, \
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
        #self.critic = Critic(self.state_dim, self.action_dim)
        #self.critic_target = Critic(self.state_dim, self.action_dim)
        self.reward_func = Reward()

        # Target network initialization
#        hard_updates(self.critic_target, self.critic)

        # Relay Buffer
        self.history = History()

        # Training models
        self.reward_criterion = nn.MSELoss()
        self.reward_optim = optim.Adam(self.reward_func.parameters(), reward_lr)

        self.actor_optim = AdamOptim(self.actor_lr)

    def get_action(self, state):
        state = Variable(torch.from_numpy(state).float())
        action = torch.squeeze(self.actor.forward(state))
        return action.detach().numpy()

    def random_action(self):
        return self.d_max * np.random.rand(self.action_dim)

    def update(self, batch_size, update_count):
        self.actor_lr = 0.001 * 1 / (1 + 0.1 * (update_count // 10000))
        self.reward_optim.param_groups[0]['lr'] = 0.001 * 1 / (1 + 0.1 * (update_count // 10000))
        # self.critic_optim.param_groups[0]['lr'] = 1e-3 * 1 / (1 + 0.1 * (epoch // 1000))
        states, actions, rewards, utilities = self.history.sample(batch_size)
        #actions = self.history.history['action']
        #U = self.history.history['utility'].reshape(-1)

        # Critic update
        g_samples = torch.FloatTensor(states[:,0])
        d_samples = torch.FloatTensor(actions)
        rewards_torch = torch.FloatTensor(rewards)
        reward_vals = self.reward_func.forward(g_samples, d_samples)
        reward_loss = self.reward_criterion(reward_vals, rewards_torch)
        self.reward_optim.zero_grad()
        reward_loss.backward()
        self.reward_optim.step()

        self.reward_func.fcz0.weight.data = torch.relu(self.reward_func.fcz0.weight.data)
        self.reward_func.fcz1.weight.data = torch.relu(self.reward_func.fcz1.weight.data)
        self.reward_func.fcz2.weight.data = torch.relu(self.reward_func.fcz2.weight.data)

        d_samples = Variable(d_samples, requires_grad = True)
        policy_loss = -self.reward_func.forward(g_samples, d_samples).mean()
        policy_loss.backward()

        dr = d_samples.grad.view(1,-1).detach().numpy()
        J_plus, J_minus = self.actor.policy_grad(states)
        d_plus_grad = np.sum(np.matmul(dr, J_plus).reshape(-1, 2), axis=0)
        d_minus_grad = np.sum(np.matmul(dr, J_minus).reshape(-1, 2), axis=0)
        # self.actor.d_plus = np.clip(self.actor_optim.update(update_count, self.actor.d_plus, d_plus_grad), 0, self.d_max)
        # self.actor.d_minus = np.clip(self.actor_optim.update(update_count, self.actor.d_minus, d_minus_grad), self.actor.d_plus, self.d_max)
        # self.actor.d_plus = np.clip(self.actor.d_plus + self.actor_lr * d_plus_grad, 0, self.d_max)
        self.actor.d_plus = self.actor.d_plus - self.actor_lr * d_plus_grad
        # self.actor.d_minus = np.clip(self.actor.d_minus + self.actor_lr * d_minus_grad, self.actor.d_plus, self.d_max)
        self.actor.d_minus = self.actor.d_minus - self.actor_lr * d_minus_grad
