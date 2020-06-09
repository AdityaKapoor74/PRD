import numpy as np
import torch 
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
from torch.autograd import Variable
from torch.distributions import Categorical
from a2c import *

class A2CAgent:

  def __init__(self,env,value_lr=2e-4, policy_lr=2e-4, gamma=0.99):
    self.env = env
    self.value_lr = value_lr
    self.policy_lr = policy_lr
    self.gamma = gamma

    self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    self.num_agents = self.env.n

    self.value_input_dim = self.env.observation_space[0].shape[0]+(self.num_agents-1)
    self.value_output_dim = 1
    self.policy_input_dim = self.env.observation_space[0].shape[0]
    self.policy_output_dim = self.env.action_space[0].n

    self.value_network = ValueNetwork(self.value_input_dim,self.value_output_dim).to(self.device)
    self.policy_network = PolicyNetwork(self.policy_input_dim,self.policy_output_dim).to(self.device)

    self.MSELoss = nn.MSELoss()
    self.value_optimizer = optim.Adam(self.value_network.parameters(),lr=self.value_lr)
    self.policy_optimizer = optim.Adam(self.policy_network.parameters(),lr=self.policy_lr)

    self.entropy_list = []
    self.value_loss_list = []
    self.policy_loss_list = []

  def get_action(self,state):
    state = torch.FloatTensor(state).to(self.device)
    logits = self.policy_network.forward(state)
    dist = F.softmax(logits,dim=0)
    # print('dist: ', dist)
    probs = Categorical(dist)

    index = probs.sample().cpu().detach().item()

    return index

  def compute(self,input_to_policy_net,next_input_to_policy_net,global_actions_batch,rewards,input_to_value_net,next_input_to_value_net):
    
    #update critic (value_net)
    curr_Q = self.value_network.forward(input_to_value_net)
    next_Q = self.value_network.forward(next_input_to_value_net)
    estimated_Q = rewards+self.gamma*next_Q
    critic_loss = self.MSELoss(curr_Q,estimated_Q.detach())


    curr_logits = self.policy_network.forward(input_to_policy_net)
    dists = F.softmax(curr_logits,dim=1)
    probs = Categorical(dists)

    entropy = []
    for dist in dists:
      entropy.append(-torch.sum(dist*torch.log(dist)))
    entropy = torch.stack(entropy).mean()
    self.entropy_list.append(entropy)

    # print('self.entropy_list: ', self.entropy_list)

    advantage = estimated_Q - curr_Q
    policy_loss = -probs.log_prob(global_actions_batch.view(global_actions_batch.size(0))).view(-1, 1) * advantage.detach()
    policy_loss = policy_loss.mean() - 0.001*entropy

    self.policy_loss_list.append(policy_loss)
    self.value_loss_list.append(critic_loss)
    
    # torch.nn.utils.clip_grad_norm_(self.actorcritic.parameters(),500)

    return critic_loss,policy_loss




  def update(self,input_to_policy_net,next_input_to_policy_net,global_actions_batch,rewards,input_to_value_net,next_input_to_value_net):
    
    #update critic (value_net)
    value_loss,policy_loss = self.compute(input_to_policy_net,next_input_to_policy_net,global_actions_batch,rewards,input_to_value_net,next_input_to_value_net)
    
    self.value_optimizer.zero_grad()
    value_loss.backward(retain_graph=True)
    self.value_optimizer.step()


    self.policy_optimizer.zero_grad()
    policy_loss.backward()
    self.policy_optimizer.step()

    # torch.nn.utils.clip_grad_norm_(self.actorcritic.parameters(),500)
