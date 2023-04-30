import torch
import numpy as np
import numpy.random as npr
from model import *
import torch.autograd
import torch.optim as optim

actor_net = Actor(3, 2, 1)
critic_net = Critic(3, 2)

a1 = torch.clone(actor_net.fc1.weight)
a2 = torch.clone(actor_net.fc2.weight)
a3 = torch.clone(actor_net.fc3.weight)
a4 = torch.clone(actor_net.fc1_th.weight)
a5 = torch.clone(actor_net.fc2_th.weight)
a6 = torch.clone(actor_net.fc3_th.weight)
x = torch.FloatTensor(npr.rand(100,3))

actor_optim = optim.Adam(actor_net.parameters(), lr = 1e-3)
policy_loss = -critic_net.forward(x, actor_net.forward(x)).mean()
actor_optim.zero_grad()
policy_loss.backward()
actor_optim.step()

a_p1 = actor_net.fc1.weight
a_p2 = actor_net.fc2.weight
a_p3 = actor_net.fc3.weight
a_p4 = actor_net.fc1_th.weight
a_p5 = actor_net.fc2_th.weight
a_p6 = actor_net.fc3_th.weight
print("{:e}".format(torch.max(torch.abs(a1 - a_p1))))
print("{:e}".format(torch.max(torch.abs(a2 - a_p2))))
print("{:e}".format(torch.max(torch.abs(a3 - a_p3))))
print("{:e}".format(torch.max(torch.abs(a4 - a_p4))))
print("{:e}".format(torch.max(torch.abs(a5 - a_p5))))
print("{:e}".format(torch.max(torch.abs(a6 - a_p6))))


'''
pi = x[:,1:]

thresholds = torch.squeeze(threshold_net.forward(pi))
nz_actions = torch.squeeze(actor_net.forward(x))

action = (x[:,0] < torch.sum(thresholds[:,[0,2]],1)).view(-1,1) * thresholds[:,[0,2]] + \
         (x[:,0] > torch.sum(thresholds[:,[1,3]],1)).view(-1,1) * thresholds[:,[1,3]] + \
         ((x[:,0] > torch.sum(thresholds[:,[0,2]],1)) * (x[:,0] < torch.sum(thresholds[:,[1,3]],1))).view(-1,1) * nz_actions

actor_optim = optim.Adam(threshold_net.parameters(), 1)
threshold_optim = optim.Adam(threshold_net.parameters(), lr =1)

current_pol_action = Variable(action, requires_grad = True)
a = actor_net.fc1.weight
policy_loss = -critic_net.forward(x, current_pol_action).mean()
actor_optim.zero_grad()
threshold_optim.zero_grad()
policy_loss.backward()
threshold_optim.step()
actor_optim.step()
a_p = actor_net.fc1.weight
'''