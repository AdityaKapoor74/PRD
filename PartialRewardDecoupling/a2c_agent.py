import numpy as np
import torch 
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
from torch.autograd import Variable
from torch.distributions import Categorical
from a2c import *
import torch.nn.functional as F
import gc
import os

class A2CAgent:

  def __init__(self,env,value_lr=1e-5, policy_lr=1e-5, gamma=0.99):
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

    
    # model_path_value = "/home/aditya/Desktop/Partial_Reward_Decoupling/git/PRD/PartialRewardDecoupling/models/one_agent/value_network_no_comms_discounted_rewards_smaller_agents_0.15_wo_termination_lr_2e-4_policy_lr_2e-4_with_grad_norm_0.5_entropy_pen_0.001_xavier_uniform_init_clamp_logs.pt"
    # model_path_policy = "/home/aditya/Desktop/Partial_Reward_Decoupling/git/PRD/PartialRewardDecoupling/models/one_agent/policy_network_no_comms_discounted_rewards_smaller_agents_0.15_wo_termination_lr_2e-4_value_lr_2e-4_with_grad_norm_0.5_entropy_pen_0.001_xavier_uniform_init_clamp_logs.pt"

    self.value_network = ValueNetwork(self.value_input_dim,self.value_output_dim).to(self.device)
    self.policy_network = PolicyNetwork(self.policy_input_dim,self.policy_output_dim).to(self.device)
    
    # self.value_network.load_state_dict(torch.load(model_path_value,map_location=torch.device('cpu')))
    # self.policy_network.load_state_dict(torch.load(model_path_policy,map_location=torch.device('cpu')))

    self.value_optimizer = optim.Adam(self.value_network.parameters(),lr=self.value_lr)
    self.policy_optimizer = optim.Adam(self.policy_network.parameters(),lr=self.policy_lr)
    

  def get_action(self,state):
    state = torch.FloatTensor(state).to(self.device)
    logits = self.policy_network.forward(state)
    dist = F.softmax(logits,dim=0)
    probs = Categorical(dist)

    index = probs.sample().cpu().detach().item()

    return index

  def update(self,input_to_policy_net,global_actions_batch,rewards,input_to_value_net):
    
    #update critic (value_net)
    curr_Q = self.value_network.forward(input_to_value_net)
    discounted_rewards = np.asarray([[torch.sum(torch.FloatTensor([self.gamma**i for i in range(rewards[k][j:].size(0))])* rewards[k][j:]) for j in range(rewards.size(0))] for k in range(self.num_agents)])
    discounted_rewards = np.transpose(discounted_rewards)
    value_targets = rewards + torch.FloatTensor(discounted_rewards).to(self.device)
    value_targets = value_targets.unsqueeze(dim=-1)
    value_loss = F.smooth_l1_loss(curr_Q,value_targets)

    #update actor (policy net)
    curr_logits = self.policy_network.forward(input_to_policy_net)
    dists = F.softmax(curr_logits,dim=-1)
    probs = Categorical(dists)

    entropy = -torch.mean(torch.sum(dists * torch.log(torch.clamp(dists, 1e-10,1.0)), dim=2))
    # print("ENTROPY:",entropy)

    advantage = value_targets - curr_Q
    policy_loss = -probs.log_prob(global_actions_batch).unsqueeze(dim=-1) * advantage.detach()
    policy_loss = policy_loss.mean() - 0.001*entropy

    
    self.value_optimizer.zero_grad()
    value_loss.backward(retain_graph=False)
    grad_norm_value = torch.nn.utils.clip_grad_norm_(self.value_network.parameters(),0.5)
    self.value_optimizer.step()
    
    self.policy_optimizer.zero_grad()
    policy_loss.backward(retain_graph=False)
    grad_norm_policy = torch.nn.utils.clip_grad_norm_(self.policy_network.parameters(),0.5)
    self.policy_optimizer.step()

    # print("SHAPES")
    # print("VALUE:",curr_Q.shape)
    # print("DISCOUNTED REWARDS:",discounted_rewards.shape)
    # print("VALUE TARGET:",value_targets.shape)
    # print("CURRENT LOGITS:",curr_logits.shape)
    # print("ACTIONS:",global_actions_batch.shape)


    return value_loss,policy_loss,entropy,grad_norm_value,grad_norm_policy
