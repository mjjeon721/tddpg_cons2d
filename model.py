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
class Actor(nn.Module) :
    def __init__(self,states_dim, action_dim, d_max, d_plus, d_minus, hidden1 = 128, hidden2 = 64, init_w = 3e-3):
        # State : Price & Renewable generation
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(states_dim, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.fc3 = nn.Linear(hidden2, action_dim)
        self.relu = nn.ReLU()
        self.d_max = d_max
        self.d_plus = torch.Tensor(d_plus)
        self.d_minus = torch.Tensor(d_minus)
        self.action_dim = action_dim
        self.sigmoid = nn.Sigmoid()
        self.init_weights(init_w)

    def init_weights(self, init_w):
        self.fc1.weight.data = fanin_init(self.fc1.weight.data.size())
        self.fc2.weight.data = fanin_init(self.fc2.weight.data.size())
        self.fc3.weight.data.uniform_(-init_w, init_w)

    def forward(self, state):
        state = state.view(-1,4)
        state_n = state[:,:3]
        ix = state[:,3].detach().numpy().astype(int)
        nz = self.fc1(state_n)
        nz = self.relu(nz)
        nz = self.fc2(nz)
        nz = self.relu(nz)
        nz = self.fc3(nz)
        #nz = self.sigmoid(nz).view(-1,self.action_dim)
        nz = torch.exp(nz).view(-1, self.action_dim)
        nz_out = nz / torch.sum(nz, 1, keepdim=True) * state_n.view(-1,3)[:,0].view(-1,1)

        th_out = torch.cat((self.d_plus[:,ix].view(-1,2), self.d_minus[:,ix].view(-1,2)), dim = 1)

        return (state[:,0] < torch.sum(th_out[:,:2],1)).view(-1,1) * th_out[:,:2] + \
         (state[:,0] > torch.sum(th_out[:,2:],1)).view(-1,1) * th_out[:,2:] + \
         ((state[:,0] > torch.sum(th_out[:,:2],1)) * (state[:,0] < torch.sum(th_out[:,2:],1))).view(-1,1) * nz_out


