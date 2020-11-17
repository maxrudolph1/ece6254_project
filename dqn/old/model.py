import torch
import torch.nn as nn
import torch.nn.functional as F

class DQN(nn.Module):

    def __init__(self, N_STATES, N_ACTIONS):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(N_STATES, N_STATES*2)
        #self.fc1.weight.data.normal_(0, 0.1)   # initialization
        self.out = nn.Linear(N_STATES*2, N_ACTIONS)
        #self.out.weight.data.normal_(0, 0.1)   # initialization

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        actions_value = self.heaviside(self.out(x))
        return actions_value