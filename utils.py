import numpy as np
from collections import deque
import random

class AdamOptim():
    def __init__(self, lr, beta1 = 0.9, beta2 = 0.999, epsilon = 1e-8):
        self.m_dw, self.v_dw = 0, 0
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.lr = lr

    def update(self, t, w, dw):

        ## momentum beta 1
        # *** thresholds *** #
        self.m_dw = self.beta1*self.m_dw + (1-self.beta1)*dw

        ## rms beta 2
        # *** thresholds *** #
        self.v_dw = self.beta2 * self.v_dw + (1 - self.beta2) * (dw ** 2)

        ## bias correction
        m_dw_corr = self.m_dw / (1 - self.beta1 ** t)
        v_dw_corr = self.v_dw / (1 - self.beta2 ** t)

        ## Update thresholds
        w = w - self.lr * (m_dw_corr / (np.sqrt(v_dw_corr) + self.epsilon))

        return w


# Buffer class
class Memory:
    def __init__(self, max_size):
        self.max_size = max_size
        self.buffer = deque(maxlen=self.max_size)

    def push(self, state, action, reward, next_state, done):
        experience = (state, action, np.array([reward]), next_state, done)
        self.buffer.append(experience)

    def sample(self, batch_size):
        state_batch = []
        action_batch = []
        reward_batch = []
        next_state_batch = []
        done_batch = []

        batch = random.sample(self.buffer, batch_size)

        for experience in batch:
            state, action, reward, next_state, done = experience
            state_batch.append(state)
            action_batch.append(action)
            reward_batch.append(reward)
            next_state_batch.append(next_state)
            done_batch.append(done)

        return state_batch, action_batch, reward_batch, next_state_batch, done_batch

    def __len__(self):
        return len(self.buffer)


def to_numpy(var):
    return var.cpu().data.numpy() if USE_CUDA else var.data.numpy()


def to_tensor(ndarray, volatile=False, requires_grad=False, dtype=float):
    return Variable(
        torch.from_numpy(ndarray), volatile=volatile, requires_grad=requires_grad
    ).type(dtype)


def soft_updates(target, source,tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(
            target_param.data * (1.0 - tau) + tau * param.data
        )

def hard_updates(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)
