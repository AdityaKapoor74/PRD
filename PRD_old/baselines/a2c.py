import torch
import torch.nn as nn
import torch.nn.functional as F

#**********************************************
# Separate Actor-Critic Network
#**********************************************
'''
Value Network
Input: Observations
Output: State-Action value

Policy Network
Input: Observations
Output: Probability over action space

(Can have separate optimizers)
'''

class ValueNetwork(nn.Module):

    def __init__(self,input_dim,output_dim):
        super(ValueNetwork,self).__init__()
        self.fc1 = nn.Linear(input_dim,512)
        torch.nn.init.xavier_uniform_(self.fc1.weight)
        self.fc2 = nn.Linear(512,256)
        torch.nn.init.xavier_uniform_(self.fc2.weight)
        self.value = nn.Linear(256,output_dim)
        torch.nn.init.xavier_uniform_(self.value.weight)

    def forward(self,state):
        value = F.relu(self.fc1(state))
        value = F.relu(self.fc2(value))
        value = self.value(value)

        return value


class PolicyNetwork(nn.Module):

    def __init__(self,input_dim,output_dim):
        super(PolicyNetwork,self).__init__()
        self.fc1 = nn.Linear(input_dim,512)
        torch.nn.init.xavier_uniform_(self.fc1.weight)
        self.fc2 = nn.Linear(512,256)
        torch.nn.init.xavier_uniform_(self.fc2.weight)
        self.policy = nn.Linear(256,output_dim)
        torch.nn.init.xavier_uniform_(self.policy.weight)

    def forward(self,state):
        logits = F.relu(self.fc1(state))
        logits = F.relu(self.fc2(logits))
        logits = self.policy(logits)
        logits = F.softmax(logits,dim=-1)


        return logits