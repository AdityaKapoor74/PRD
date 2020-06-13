#import matplotlib
#import matplotlib.pyplot as plt

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
    actions = torch.LongTensor([sars[1] for sars in trajectory]).view(-1, 1).to(self.device)
    rewards = torch.FloatTensor([sars[2] for sars in trajectory]).to(self.device)
    next_states = torch.FloatTensor([sars[3] for sars in trajectory]).to(self.device)
    dones = torch.FloatTensor([sars[4] for sars in trajectory]).view(-1, 1).to(self.device)

    self.agents.update(states,next_states,actions,rewards,episode)


  def run(self,max_episode,max_steps):
    for episode in range(max_episode):
      states = self.env.reset()
      trajectory = []
      episode_reward = 0
      for step in range(max_steps):
        actions = self.get_actions(states)
        # print(actions)
        next_states,rewards,dones,_ = self.env.step(actions)
        episode_reward += np.mean(rewards)
        

        if all(dones) or step == max_steps-1:
          dones = [1 for _ in range(self.num_agents)]
          sarsd = [[states[i],actions[i].argmax(),rewards[i],next_states[i],dones[i]] for i in range(len(states))]
          for i in sarsd:
            trajectory.append(i)
          print("*"*200)
          print("EPISODE: {} | REWARD: {} \n".format(episode,np.round(episode_reward,decimals=4)))
          print("*"*200)
          self.agents.writer.add_scalar('Lenght of the episode',step,episode)
          self.episode_rewards.append(episode_reward)
          self.agents.writer.add_scalar('Reward',self.episode_rewards[-1],episode)
          break
        else:
          dones = [0 for _ in range(self.num_agents)]
          sarsd = [[states[i],actions[i].argmax(),rewards[i],next_states[i],dones[i]] for i in range(len(states))]
          for i in sarsd:
            trajectory.append(i)
          states = next_states
      
#       make a directory called models
      if episode%500:
        torch.save(self.agents.actorcritic, "./models/actorcritic_network")
      
        
      self.update(trajectory,episode)

#     return episode_rewards,self.agents.entropy_list,self.agents.value_loss_list,self.agents.policy_loss_list,self.agents.total_loss_list

