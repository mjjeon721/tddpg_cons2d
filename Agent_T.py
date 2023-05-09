from model import *
from utils import *
import torch.optim as optim

class TDDPGAgent:
    def __init__(self, state_dim, action_dim, actor_lr = 0.001, critic_lr=0.001, \
                 eta=1, d_max = 1, tau=0.001, max_memory_size = 10000):
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
        self.memory = Memory(max_memory_size)

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

    def update(self, batch_size, update_count):
        self.actor_optim.lr = 0.001 * 1 / (1 + 0.1 * (update_count // 5000))
        #self.critic_optim.param_groups[0]['lr'] = 1e-3 * 1 / (1 + 0.1 * (epoch // 1000))
        states, actions, rewards, next_states, dones = self.memory.sample(batch_size)
        states = torch.FloatTensor(np.vstack(states))
        actions = torch.FloatTensor(np.vstack(actions))
        rewards = torch.FloatTensor(np.vstack(rewards))
        next_states = torch.FloatTensor(np.vstack(next_states))
        dones = torch.Tensor(np.array(dones)).view(-1,1)

        Qvals = self.critic.forward(states, actions)
        next_actions = self.actor.forward(next_states)

        next_Q = self.critic_target.forward(next_states, next_actions)
        Qprime = rewards + self.eta * next_Q
        critic_loss = self.critic_criterion(Qvals, Qprime)

        # Critic updates
        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()

        # Policy updates
        current_pol_actions = Variable(self.actor.forward(states), requires_grad = True)
        self.critic_optim.zero_grad()
        policy_loss = -self.critic.forward(states, current_pol_actions).mean()

        policy_loss.backward()
        J_plus, J_minus = self.actor.policy_grad(states)
        Q_grad = current_pol_actions.grad.view(1,-1).detach().numpy()
        d_plus_grad = np.sum(np.matmul(Q_grad, J_plus).reshape(-1,2), axis = 0)
        d_minus_grad = np.sum(np.matmul(Q_grad, J_minus).reshape(-1,2), axis = 0)
        self.actor.d_plus = np.clip(self.actor_optim.update(update_count, self.actor.d_plus, d_plus_grad), 0, self.d_max)
        self.actor.d_minus = np.clip(self.actor_optim.update(update_count, self.actor.d_minus, d_minus_grad), self.actor.d_plus, self.d_max)
        #self.actor.d_plus = np.clip(self.actor.d_plus - self.actor_lr * np.sum(np.matmul(Q_grad, J_plus).reshape(-1,2), axis = 0), 0, self.d_max)
        #self.actor.d_minus = np.clip(self.actor.d_minus - self.actor_lr * np.sum(np.matmul(Q_grad, J_minus).reshape(-1, 2), axis=0), self.actor.d_plus, self.d_max)
        soft_updates(self.critic_target, self.critic, self.tau)