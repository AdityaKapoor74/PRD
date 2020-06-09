import torch
import torch.nn as nn
import torch.nn.functional as F

# Centralized Policy Value Network
class CentralizedActorCritic(nn.Module): 

    def __init__(self, obs_dim, action_dim):
        super(CentralizedActorCritic, self).__init__()

        self.obs_dim = obs_dim
        self.action_dim = action_dim

        self.value1 = nn.Linear(self.obs_dim, 256)
        self.value2 = nn.Linear(256, 1)

        self.policy1 = nn.Linear(self.obs_dim, 256)
        self.policy2 = nn.Linear(256, self.action_dim)

    def forward(self, x):
        x_v = F.relu(self.value1(x))
        qval = self.value2(x_v)

        x_p = F.relu(self.policy1(x))
        policy = self.policy2(x_p)

        return policy,qval

