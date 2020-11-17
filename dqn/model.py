import torch
import torch.nn as nn
import torch.nn.functional as F

class DQN(nn.Module):

    def __init__(self, N_STATES, N_ACTIONS):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(N_STATES, N_STATES*4)
        #self.fc1.weight.data.normal_(0, 0.1)   # initialization
        self.fc2 = nn.Linear(N_STATES*4, N_STATES*4)
        self.out = nn.Linear(N_STATES*4, N_ACTIONS)

        self.final = nn.Sigmoid()
        #self.out.weight.data.normal_(0, 0.1)   # initialization

    def forward(self, x):
        x = F.relu(self.fc1(x))

        x = F.relu(self.fc2(x))
   


        actions_value = self.out(x)
        return actions_value