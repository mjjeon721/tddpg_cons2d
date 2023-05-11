import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd
from torch.autograd import Variable

# Weight initialization function
def fanin_init(size, fanin = None) :
    fanin = fanin or size[0]
    v = 1. / np.sqrt(fanin)
    return torch.Tensor(size).uniform_(-v, v)

# Critic network class
class Critic(nn.Module) :
    def __init__(self, states_dim, action_dim, hidden1 = 128, hidden2 = 64, init_w = 3e-4):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(states_dim, hidden1)
        self.fc2 = nn.Linear(hidden1 + action_dim, hidden2)
        self.fc3 = nn.Linear(hidden2, 1)
        self.relu = nn.ReLU()
        self.init_weights(init_w)
        #Batch Normalization
        self.BN0 = nn.BatchNorm1d(states_dim)
        self.BN1 = nn.BatchNorm1d(hidden1)

    def init_weights(self, init_w):
        self.fc1.weight.data = fanin_init(self.fc1.weight.data.size())
        self.fc2.weight.data = fanin_init(self.fc2.weight.data.size())
        self.fc3.weight.data.uniform_(-init_w, init_w)

    def forward(self, state, action):
        x = state[:,:3]
        a = action
        # Batch normalization
        x = self.BN0(x)
        out = self.fc1(x)
        out = self.relu(out)
        # Batch normalization
        out = self.BN1(out)
        out = self.fc2(torch.cat((out,a), dim = 1))
        out = self.relu(out)
        out = self.fc3(out)
        return out

# Actor network class
class Actor:
    def __init__(self,states_dim, action_dim, d_max):
        # State : Price & Renewable generation
        self.d_max = d_max
        self.action_dim = action_dim
        # threshold initialization
        self.d_plus = 0.5 * self.d_max * np.random.rand(2)
        self.d_minus = 0.5 * self.d_max * np.random.rand(2) + 0.5 * self.d_max
        #self.d_plus, self.d_minus = self.d_max * np.sort(np.random.rand(4)).reshape(2,-1)
        self.d_plus = self.d_plus.reshape(1, self.action_dim)
        self.d_minus = self.d_minus.reshape(1, self.action_dim)

    def forward(self, state):
        state_np = state.detach().numpy()
        state_np = state_np.reshape(-1, 3)

        nz_actions = self.d_plus + (state_np[:,0] - np.sum(self.d_plus)).reshape(-1,1) * (self.d_minus - self.d_plus) / \
                     (np.sum(self.d_minus) - np.sum(self.d_plus))
        actions = (state_np[:,0] < np.sum(self.d_plus)).reshape(-1,1) * self.d_plus\
                  + (state_np[:,0] > np.sum(self.d_minus)).reshape(-1,1) * self.d_minus \
                  + ((state_np[:, 0] > np.sum(self.d_plus)) * (state_np[:,0] < np.sum(self.d_minus))).reshape(-1,1) * nz_actions

        return torch.FloatTensor(actions)

    def policy_grad(self, state):
        state_np = state.detach().numpy()
        m = state_np.shape[0]
        J_plus = np.empty((0,self.action_dim))
        J_minus = np.empty((0, self.action_dim))
        for g_t in state_np[:,0] :
            if g_t < np.sum(self.d_plus) :
                J_plus = np.vstack((J_plus, np.eye(self.action_dim)))
                J_minus = np.vstack((J_minus, np.zeros((self.action_dim, self.action_dim))))
            elif g_t > np.sum(self.d_minus) :
                J_plus = np.vstack((J_plus, np.zeros((self.action_dim, self.action_dim))))
                J_minus = np.vstack((J_minus, np.eye(self.action_dim)))
            else :
                A = np.zeros((self.action_dim, self.action_dim))
                B = np.zeros((self.action_dim, self.action_dim))
                nab_ii = 1 - (self.d_minus - self.d_plus) / (np.sum(self.d_minus) - np.sum(self.d_plus)) \
                + (g_t - np.sum(self.d_plus)) * (self.d_minus - self.d_plus + np.sum(self.d_plus) - np.sum(self.d_minus)) \
                / (np.sum(self.d_plus) - np.sum(self.d_minus)) ** 2
                nab_ii_m = (g_t - np.sum(self.d_plus)) * (-self.d_minus + self.d_plus - np.sum(self.d_plus) + np.sum(self.d_minus)) \
                         / (np.sum(self.d_plus) - np.sum(self.d_minus)) ** 2
                nab_ii = nab_ii.reshape(-1)
                nab_ii_m = nab_ii_m.reshape(-1)
                A = A + np.diag(nab_ii)
                B = B + np.diag(nab_ii_m)
                nab_ij = - (self.d_minus - self.d_plus) / (np.sum(self.d_minus) - np.sum(self.d_plus)) \
                        + (g_t - np.sum(self.d_plus)) * (self.d_minus - self.d_plus) / (np.sum(self.d_plus) - np.sum(self.d_minus)) ** 2
                nab_ij_m = (g_t - np.sum(self.d_plus)) * (-self.d_minus + self.d_plus) / (np.sum(self.d_plus) - np.sum(self.d_minus)) ** 2
                nab_ij = nab_ij.reshape(-1)
                nab_ij_m = nab_ij_m.reshape(-1)
                A = A + np.fliplr(np.diag(nab_ij))
                B = B + np.fliplr(np.diag(nab_ij_m))
                J_plus = np.vstack((J_plus, A))
                J_minus = np.vstack((J_minus, B))
        diag_matrix_plus = np.zeros((self.action_dim * m, self.action_dim * m))
        diag_matrix_minus= np.zeros((self.action_dim * m, self.action_dim * m))
        for i in range(m) :
            block_start = i * self.action_dim
            block_end = (i+1) * self.action_dim
            block = J_plus[i * self.action_dim:(i+1) * self.action_dim,:]
            block_m = J_minus[i * self.action_dim:(i+1) * self.action_dim,:]
            diag_matrix_plus[block_start:block_end, block_start:block_end] = block
            diag_matrix_minus[block_start:block_end, block_start:block_end] = block_m

        return diag_matrix_plus, diag_matrix_minus

