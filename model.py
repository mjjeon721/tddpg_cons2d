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
        x = state
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
    def __init__(self,states_dim, action_dim, d_max, hidden1 = 128, hidden2 = 64, init_w = 3e-3):
        # State : Price & Renewable generation
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(states_dim, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.fc3 = nn.Linear(hidden2, action_dim)
        self.relu = nn.ReLU()
        self.d_max = d_max
        self.action_dim = action_dim
        self.sigmoid = nn.Sigmoid()
        self.fc1_th = nn.Linear(2, hidden1)
        self.fc2_th = nn.Linear(hidden1, hidden2)
        self.fc3_th = nn.Linear(hidden2, action_dim * 2)
        self.init_weights(init_w)

    def init_weights(self, init_w):
        self.fc1.weight.data = fanin_init(self.fc1.weight.data.size())
        self.fc2.weight.data = fanin_init(self.fc2.weight.data.size())
        self.fc3.weight.data.uniform_(-init_w, init_w)
        self.fc1_th.weight.data = fanin_init(self.fc1_th.weight.data.size())
        self.fc2_th.weight.data = fanin_init(self.fc2_th.weight.data.size())
        self.fc3_th.weight.data.uniform_(-init_w, init_w)
    def forward(self, state):
        state = state.view(-1,3)
        nz = self.fc1(state)
        nz = self.relu(nz)
        nz = self.fc2(nz)
        nz = self.relu(nz)
        nz = self.fc3(nz)
        nz = self.sigmoid(nz).view(-1,self.action_dim)
        nz_out = nz / torch.sum(nz, 1, keepdim=True) * state.view(-1,3)[:,0].view(-1,1)

        th = self.fc1_th(state[:,1:])
        th = self.relu(th)
        th = self.fc2_th(th)
        th = self.relu(th)
        th = self.fc3_th(th)
        th = self.sigmoid(th).view(-1, self.action_dim * 2)
        th_out = torch.cat((torch.sort(th[:, :2], dim=1)[0], torch.sort(th[:, 2:], dim=1)[0]), dim=1)

        return (state[:,0] < torch.sum(th_out[:,[0,2]],1)).view(-1,1) * th_out[:,[0,2]] + \
         (state[:,0] > torch.sum(th_out[:,[1,3]],1)).view(-1,1) * th_out[:,[1,3]] + \
         ((state[:,0] > torch.sum(th_out[:,[0,2]],1)) * (state[:,0] < torch.sum(th_out[:,[1,3]],1))).view(-1,1) * nz_out


class Threshold(nn.Module) :
    def __init__(self, states_dim, action_dim, d_max, hidden1 = 400, hidden2 = 300, init_w = 3e-3):
        super(Threshold, self).__init__()
        self.fc1 = nn.Linear(states_dim, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.fc3 = nn.Linear(hidden2, action_dim * 2)
        self.relu = nn.ReLU()
        self.d_max = d_max
        self.sigmoid = nn.Sigmoid()
        self.action_dim = action_dim
        self.init_weights(init_w)

    def init_weights(self, init_w):
        self.fc1.weight.data = fanin_init(self.fc1.weight.data.size())
        self.fc2.weight.data = fanin_init(self.fc2.weight.data.size())
        self.fc3.weight.data.uniform_(-init_w, init_w)

    def forward(self, state):
        # Return threshold values,
        out = self.fc1(state)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        out = self.sigmoid(out).view(-1,self.action_dim * 2)
        return torch.cat((torch.sort(out[:,:2], dim = 1)[0], torch.sort(out[:,2:], dim = 1)[0]), dim = 1)
