#import matplotlib
#import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F 
import torch.optim as optim
from torch.distributions import Categorical
import torch.autograd as autograd
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from a2c_agent import A2CAgent
import gc


class MAA2C:

  def __init__(self,env):
    self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    self.env = env
    self.num_agents = env.n
    self.agents = A2CAgent(self.env)


    self.writer = SummaryWriter('longer_runs/one_agent/simple_spread_1_hidden_layer_no_comms_discounted_rewards_smaller_agents_0.15_wo_termination_value_lr_2e-4_policy_lr_2e-4_grad_norm_0.5_entropy_pen_0.001_xavier_uniform_init_clamp_logs')

  def get_actions(self,states):
    actions = []
    for i in range(self.num_agents):
      action = self.agents.get_action(states[i])
      actions.append(action)
    return actions

  def update(self,trajectory,episode):

    states_policy = torch.FloatTensor([sars[0] for sars in trajectory]).to(self.device)
    actions = torch.LongTensor([sars[1] for sars in trajectory]).to(self.device)
    rewards = torch.FloatTensor([sars[2] for sars in trajectory]).to(self.device)
    states_value = torch.FloatTensor([sars[3] for sars in trajectory]).to(self.device)


    value_loss,policy_loss,entropy,grad_norm_value,grad_norm_policy = self.agents.update(states_policy,actions,rewards,states_value)

    self.writer.add_scalar('Loss/Entropy loss',entropy,episode)
    self.writer.add_scalar('Loss/Value Loss',value_loss,episode)
    self.writer.add_scalar('Loss/Policy Loss',policy_loss,episode)
    self.writer.add_scalar('Gradient Normalization/Grad Norm Value',grad_norm_value,episode)
    self.writer.add_scalar('Gradient Normalization/Grad Norm Policy',grad_norm_policy,episode)




  def run(self,max_episode,max_steps):  
    for episode in range(max_episode):
      states_policy = self.env.reset()
      trajectory = []
      episode_reward = 0
      for step in range(max_steps):
        actions = self.get_actions(states_policy)
        next_states,rewards,dones,info = self.env.step(actions)

        states_value = []

        for i in range(self.num_agents):
          actions_copy = np.copy(actions)
          actions_copy = np.delete(actions_copy,[i])
          tmp = np.copy(states_policy[i])
          for j in range(self.num_agents-1):
            if j==self.num_agents-2:
              tmp = np.append(tmp,actions_copy[j])
            else:
              tmp = np.insert(tmp,-(self.num_agents-2-j),actions_copy[j])
          # states_value.append(np.append(states_policy[i],actions_copy))
          states_value.append(tmp)

        states_value = np.asarray(states_value)

        episode_reward += np.sum(rewards)


        if all(dones) or step == max_steps-1:

          dones = [1 for _ in range(self.num_agents)]
          trajectory.append([states_policy,actions,rewards,states_value])
          print("*"*100)
          print("EPISODE: {} | REWARD: {} \n".format(episode,np.round(episode_reward,decimals=4)))
          print("*"*100)
          self.writer.add_scalar('Reward Incurred/Length of the episode',step,episode)
          self.writer.add_scalar('Reward Incurred/Reward',episode_reward,episode)
          break
        else:
          dones = [0 for _ in range(self.num_agents)]
          trajectory.append([states_policy,actions,rewards,states_value])
          states = next_states
      
#       make a directory called models
      if episode%500:
        torch.save(self.agents.value_network.state_dict(), "./models/one_agent/1_hidden_layer_value_network_no_comms_discounted_rewards_smaller_agents_0.15_wo_termination_lr_2e-4_policy_lr_2e-4_with_grad_norm_0.5_entropy_pen_0.001_xavier_uniform_init_clamp_logs.pt")
        torch.save(self.agents.policy_network.state_dict(), "./models/one_agent/1_hidden_layer_policy_network_no_comms_discounted_rewards_smaller_agents_0.15_wo_termination_lr_2e-4_value_lr_2e-4_with_grad_norm_0.5_entropy_pen_0.001_xavier_uniform_init_clamp_logs.pt")      
        
      self.update(trajectory,episode) 

