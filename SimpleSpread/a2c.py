import torch
import torch.nn as nn
import torch.nn.functional as F

# Centralized Policy Value Network
class CentralizedActorCritic(nn.Module): 

    def __init__(self, obs_dim, action_dim):
        super(CentralizedActorCritic, self).__init__()

        self.obs_dim = obs_dim
        self.action_dim = action_dim

        self.shared_layer = nn.Linear(self.obs_dim, 256)
        self.value = nn.Linear(256, 1)
        self.policy = nn.Linear(256, self.action_dim)

    def forward(self, x):
        x_s = F.relu(self.shared_layer(x))
        qval = self.value(x_s)
        policy = self.policy2(x_s)

        return policy,qval

