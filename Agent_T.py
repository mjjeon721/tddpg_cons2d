from model import *
from utils import *
import torch.optim as optim

class TDDPGAgent:
    def __init__(self, state_dim, action_dim, d_plus, d_minus, actor_lr = 1e-4, critic_lr=1e-3, \
                 eta=1, d_max = 1, tau=0.001, max_memory_size = 50000):
        # Parameters
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.actor_lr = actor_lr

        self.d_plus = d_plus
        self.d_minus = d_minus

        self.eta = eta
        self.tau = tau

        self.d_max = d_max

        # Network object
        self.actor = Actor(self.state_dim, self.action_dim, self.d_max, self.d_plus, self.d_minus)
        self.actor_target = Actor(self.state_dim, self.action_dim, self.d_max, self.d_plus, self.d_minus)
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