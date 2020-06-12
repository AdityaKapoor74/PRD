#import matplotlib
#import matplotlib.pyplot as plt
import random

import torch
import torch.nn.functional as F 
import torch.optim as optim
from torch.distributions import Categorical
import torch.autograd as autograd
import numpy as np

from a2c_agent import A2CAgent


class MAA2C:

  def __init__(self,env):
    self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    self.env = env
    self.num_agents = env.n
    self.agents = A2CAgent(self.env)
    self.episode_rewards = []

  def get_actions(self,states):
    actions = []
    for i in range(self.num_agents):
      # print("MAA2C get_action")
      # print(len(states[i]))
      action = self.agents.get_action(states[i])
      actions.append(action)
    return actions

  def plot(self,rewards):
    plt.figure(2)
    plt.clf()
    plt.title('Training..')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.plot(rewards)

  def update(self,trajectory,episode):

    states = torch.FloatTensor([sars[0] for sars in trajectory]).to(self.device)
    actions = torch.FloatTensor([sars[1] for sars in trajectory]).view(-1, 1).to(self.device)
    rewards = torch.FloatTensor([sars[2] for sars in trajectory]).to(self.device)
    next_states = torch.FloatTensor([sars[3] for sars in trajectory]).to(self.device)
    dones = torch.FloatTensor([sars[4] for sars in trajectory]).view(-1, 1).to(self.device)
    inputs_to_value_net = torch.FloatTensor([sars[5] for sars in trajectory]).to(self.device)
    next_inputs_to_value_net = torch.FloatTensor([sars[6] for sars in trajectory]).to(self.device)

    self.agents.update(states,next_states,actions,rewards,inputs_to_value_net,next_inputs_to_value_net,episode)


  def run(self,max_episode,max_steps):
    
    for episode in range(max_episode):
      states = self.env.reset()

      input_to_value_net = []
      next_input_to_value_net = []
      for i in range(self.num_agents):
        input_to_value_net.append([])
        next_input_to_value_net.append([])

      trajectory = []
      episode_reward = 0
      for step in range(max_steps):

        actions = self.get_actions(states)

        for i in range(len(states)):
          action_ = np.delete(actions,i)
          input_to_value_net[i] = np.append(states[i],action_.astype(float))

        input_to_value_net = np.asarray(input_to_value_net)

        next_states,rewards,dones,_ = self.env.step(actions)

        next_actions = self.get_actions(next_states)


        for i in range(len(next_states)):
          action_ = np.delete(actions,i)
          next_input_to_value_net[i] = np.append(next_states[i],action_.astype(float))

        next_input_to_value_net = np.asarray(next_input_to_value_net)

        episode_reward += np.mean(rewards)

        if all(dones) or step == max_steps-1:
          dones = [1 for _ in range(self.num_agents)]
          sarsd = [[states[i],float(actions[i]),rewards[i],next_states[i],dones[i],input_to_value_net[i],next_input_to_value_net[i]] for i in range(len(states))]
          for i in sarsd:
            trajectory.append(i)
          print("episode: {} | reward: {} \n".format(episode,np.round(episode_reward,decimals=4)))
          break
        else:
          dones = [0 for _ in range(self.num_agents)]
          sarsd = [[states[i],float(actions[i]),rewards[i],next_states[i],dones[i],input_to_value_net[i],next_input_to_value_net[i]] for i in range(len(states))]
          for i in sarsd:
            trajectory.append(i)
          states = next_states
      
#       make a directory by the name of models
      if episode%500:
        torch.save(self.agents.value_network, "./models/value_network")
        torch.save(self.agents.policy_network,"./models/policy_network")
        
      self.episode_rewards.append(episode_reward)
      self.agents.writer.add_scalar('Reward',self.episode_rewards[-1],len(self.episode_rewards))
        
      self.update(trajectory,episode)

    # return self.episode_rewards,self.agents.entropy_list,self.agents.value_loss_list,self.agents.policy_loss_list

