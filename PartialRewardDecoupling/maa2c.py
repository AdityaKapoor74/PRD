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

  def __init__(self,env,gif=True):
    self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    self.env = env
    self.gif = gif
    self.num_agents = env.n
    self.agents = A2CAgent(self.env)

    if not(self.gif):
      # Separate network with action conditioning
      self.writer = SummaryWriter('../../runs/separate_net_with_action_conditioning/4_agents/value_lr_2e-4_policy_lr_2e-4_entropy_0.008_lambda_0.0')
      # Separate network
      # self.writer = SummaryWriter('../../runs/separate_net/4_agents/value_lr_2e-4_policy_lr_2e-4_entropy_0.008')
      # # Shared network
      # self.writer = SummaryWriter('./runs/shared_network/3_agents/value_lr_2e-4_policy_lr_2e-4_entropy_0.008')
      # # Two headed network
      # self.writer = SummaryWriter('./runs/two_headed_network/3_agents/value_lr_2e-4_policy_lr_2e-4_entropy_0.008')

      # # For comparison
      # self.writer = SummaryWriter('./runs/compare/3_agents/value_lr_2e-4_policy_lr_2e-4_entropy_0.008')

  def get_actions(self,states):
    actions = []
    for i in range(self.num_agents):
      action = self.agents.get_action(states[i])
      actions.append(action)
    return actions

  def update(self,trajectory,episode):

    states_critic = torch.FloatTensor([sars[0] for sars in trajectory]).to(self.device)
    states_actor = torch.FloatTensor([sars[1] for sars in trajectory]).to(self.device)
    next_states_critic = torch.FloatTensor([sars[2] for sars in trajectory]).to(self.device)
    next_states_actor = torch.FloatTensor([sars[3] for sars in trajectory]).to(self.device)
    actions = torch.FloatTensor([sars[4] for sars in trajectory]).to(self.device)
    rewards = torch.FloatTensor([sars[5] for sars in trajectory])

    # states = torch.FloatTensor([sars[0] for sars in trajectory]).to(self.device)
    # next_states = torch.FloatTensor([sars[1] for sars in trajectory]).to(self.device)
    # actions = torch.FloatTensor([sars[2] for sars in trajectory]).to(self.device)
    # rewards = torch.FloatTensor([sars[3] for sars in trajectory]).to(self.device)


# ***********************************************************************************
    # Separate networks with action conditioning
    value_loss,policy_loss,entropy,grad_norm_value,grad_norm_policy = self.agents.update(states_critic,states_actor,next_states_critic,next_states_actor,actions,rewards)

    # # Shared networks and Separate networks
    # value_loss,policy_loss,entropy,grad_norm = self.agents.update(states,next_states,actions,rewards)

    if not(self.gif):
      # Separate networks
      self.writer.add_scalar('Loss/Entropy loss',entropy,episode)
      self.writer.add_scalar('Loss/Value Loss',value_loss,episode)
      self.writer.add_scalar('Loss/Policy Loss',policy_loss,episode)
      self.writer.add_scalar('Gradient Normalization/Grad Norm Value',grad_norm_value,episode)
      self.writer.add_scalar('Gradient Normalization/Grad Norm Policy',grad_norm_policy,episode)

      # # Shared networks
      # self.writer.add_scalar('Loss/Entropy loss',entropy,episode)
      # self.writer.add_scalar('Loss/Value Loss',value_loss,episode)
      # self.writer.add_scalar('Loss/Policy Loss',policy_loss,episode)
      # self.writer.add_scalar('Gradient Normalization/Grad Norm Value',grad_norm,episode)


    


    
# ***********************************************************************************

  def split_states(self,states):

    states_critic = []
    states_actor = []
    for i in range(self.num_agents):
      states_critic.append(states[i][0])
      states_actor.append(states[i][1])

    states_critic = np.asarray(states_critic)
    states_actor = np.asarray(states_actor)

    return states_critic,states_actor




  def run(self,max_episode,max_steps):  
    for episode in range(1,max_episode):
      states = self.env.reset()

      states_critic,states_actor = self.split_states(states)

      trajectory = []
      episode_reward = 0
      for step in range(max_steps):

        actions = self.get_actions(states_actor)
        next_states,rewards,dones,info = self.env.step(actions)
        next_states_critic,next_states_actor = self.split_states(next_states)

        episode_reward += np.sum(rewards)


        if all(dones) or step == max_steps-1:

          dones = [1 for _ in range(self.num_agents)]
          trajectory.append([states_critic,states_actor,next_states_critic,next_states_actor,actions,rewards])
          print("*"*100)
          print("EPISODE: {} | REWARD: {} \n".format(episode,np.round(episode_reward,decimals=4)))
          print("*"*100)

          if not(self.gif):
            self.writer.add_scalar('Reward Incurred/Length of the episode',step,episode)
            self.writer.add_scalar('Reward Incurred/Reward',episode_reward,episode)
            
          break
        else:
          dones = [0 for _ in range(self.num_agents)]
          trajectory.append([states_critic,states_actor,next_states_critic,next_states_actor,actions,rewards])
          states_critic,states_actor = next_states_critic,next_states_actor
          states = next_states
      
#       make a directory called models
      if not(episode%100) and episode!=0 and not(self.gif):
#         Separate network with action conditioning
        torch.save(self.agents.value_network.state_dict(), "../../models/separate_net_with_action_conditioning/4_agents/value_net_lr_2e-4_policy_lr_2e-4_with_grad_norm_0.5_entropy_pen_0.008_xavier_uniform_init_clamp_logs_lambda_0.0.pt")
        torch.save(self.agents.policy_network.state_dict(), "../../models/separate_net_with_action_conditioning/4_agents/policy_net_lr_2e-4_value_lr_2e-4_with_grad_norm_0.5_entropy_pen_0.008_xavier_uniform_init_clamp_logs_lambda_0.0.pt")  
#         # Separate networks
        # torch.save(self.agents.value_network.state_dict(), "../../models/separate_net/4_agents/value_net_lr_2e-4_policy_lr_2e-4_with_grad_norm_0.5_entropy_pen_0.008_xavier_uniform_init_clamp_logs.pt")
        # torch.save(self.agents.policy_network.state_dict(), "../../models/separate_net/4_agents/policy_net_lr_2e-4_value_lr_2e-4_with_grad_norm_0.5_entropy_pen_0.008_xavier_uniform_init_clamp_logs.pt")  
        # # Shared network
        # torch.save(self.agents.actorcritic.state_dict(), "./models/shared_network/3_agents/value_net_lr_2e-4_policy_lr_2e-4_with_grad_norm_0.5_entropy_pen_0.008_xavier_uniform_init_clamp_logs.pt")
        # # Two headed network
        # torch.save(self.agents.actorcritic.state_dict(), "./models/two_headed_network/3_agents/value_net_lr_2e-4_policy_lr_2e-4_with_grad_norm_0.5_entropy_pen_0.008_xavier_uniform_init_clamp_logs.pt")
    


      self.update(trajectory,episode) 

