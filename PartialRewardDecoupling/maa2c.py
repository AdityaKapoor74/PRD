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

#     self.writer = SummaryWriter('/home/aditya/Desktop/Partial_Reward_Decoupling/git/PRD/PartialRewardDecoupling_new/runs/3_agents/value_lr_2e-4_policy_lr_2e-4_entropy_0.008')
    self.writer = SummaryWriter('./runs/4_agents/value_lr_2e-4_policy_lr_2e-4_entropy_0.008') # For high bay

  def get_actions(self,states):
    actions = []
    for i in range(self.num_agents):
      action = self.agents.get_action(states[i])
      actions.append(action)
    return actions

  def update(self,trajectory,episode):

    states = torch.FloatTensor([sars[0] for sars in trajectory]).to(self.device)
    next_states = torch.LongTensor([sars[1] for sars in trajectory]).to(self.device)
    actions = torch.FloatTensor([sars[2] for sars in trajectory]).to(self.device)
    rewards = torch.FloatTensor([sars[3] for sars in trajectory]).to(self.device)

# ***********************************************************************************
    value_loss,policy_loss,entropy,grad_norm_value,grad_norm_policy = self.agents.update(states,next_states,actions,rewards)
    # self.agents.update(states_policy,states_weight,actions,rewards)


    self.writer.add_scalar('Loss/Entropy loss',entropy,episode)
    self.writer.add_scalar('Loss/Value Loss',value_loss,episode)
    self.writer.add_scalar('Loss/Policy Loss',policy_loss,episode)
    self.writer.add_scalar('Gradient Normalization/Grad Norm Value',grad_norm_value,episode)
    self.writer.add_scalar('Gradient Normalization/Grad Norm Policy',grad_norm_policy,episode)
# ***********************************************************************************



  def run(self,max_episode,max_steps):  
    for episode in range(164405,max_episode):
      states = np.asarray(self.env.reset())

      trajectory = []
      episode_reward = 0
      for step in range(max_steps):

        actions = self.get_actions(states)
        next_states,rewards,dones,info = self.env.step(actions)

        episode_reward += np.sum(rewards)


        if all(dones) or step == max_steps-1:

          dones = [1 for _ in range(self.num_agents)]
          trajectory.append([states,next_states,actions,rewards])
          print("*"*100)
          print("EPISODE: {} | REWARD: {} \n".format(episode,np.round(episode_reward,decimals=4)))
          print("*"*100)
          self.writer.add_scalar('Reward Incurred/Length of the episode',step,episode)
          self.writer.add_scalar('Reward Incurred/Reward',episode_reward,episode)
          break
        else:
          dones = [0 for _ in range(self.num_agents)]
          trajectory.append([states,next_states,actions,rewards])
          states = next_states
      
#       make a directory called models
      if episode%100 and episode!=0:
        torch.save(self.agents.value_network.state_dict(), "./models/4_agents/value_net_lr_2e-4_policy_lr_2e-4_with_grad_norm_0.5_entropy_pen_0.008_xavier_uniform_init_clamp_logs.pt")
        torch.save(self.agents.policy_network.state_dict(), "./models/4_agents/policy_net_lr_2e-4_value_lr_2e-4_with_grad_norm_0.5_entropy_pen_0.008_xavier_uniform_init_clamp_logs.pt")      

      self.update(trajectory,episode) 

