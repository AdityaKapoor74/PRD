import numpy as np
import torch 
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
from torch.autograd import Variable
from torch.distributions import Categorical
from a2c import *
from torch.utils.tensorboard import SummaryWriter

class A2CAgent:

  def __init__(self,env,lr=2e-4,gamma=0.99):
    self.env = env
    self.lr = lr
    self.gamma = gamma

    self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    self.num_agents = self.env.n

    self.input_dim = env.observation_space[0].shape[0]
    self.action_dim = self.env.action_space[0].n
    self.actorcritic = CentralizedActorCritic(self.input_dim,self.action_dim).to(self.device)

    self.MSELoss = nn.MSELoss()
    self.actorcritic_optimizer = optim.Adam(self.actorcritic.parameters(),lr=lr)

    self.entropy_list = []
    self.value_loss_list = []
    self.policy_loss_list = []
    self.total_loss_list = []
    self.writer = SummaryWriter('runs/simple_spread_lr_2e-4')

  def get_action(self,state):
    state = torch.FloatTensor(state).to(self.device)
    logits,_ = self.actorcritic.forward(state)
    dist = F.softmax(logits,dim=0)
    # print('dist: ', dist)
    probs = Categorical(dist)

    index = probs.sample().cpu().detach().item()
    

    # return index

    one_hot = torch.zeros(self.action_dim)

    one_hot[int(index)] = 1
    
#     print("*"*100)
#     print("Action number:",index)
#     print("One hot vec:",one_hot)
#     print("*"*100)

    return one_hot


  def update(self,global_state_batch,global_next_state_batch,global_actions_batch,rewards,episode):
    
    #update actorcritic
    curr_logits,curr_Q = self.actorcritic.forward(global_state_batch)
    rewards = rewards.reshape(-1,1)
    _,next_Q = self.actorcritic.forward(global_next_state_batch)
    estimated_Q = rewards + self.gamma*next_Q
    

    critic_loss = self.MSELoss(curr_Q,estimated_Q.detach()).mean()
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
    policy_loss = policy_loss.mean()
    self.policy_loss_list.append(policy_loss)
    self.value_loss_list.append(critic_loss)

    total_loss = policy_loss + critic_loss - 0.1*entropy
    self.total_loss_list.append(total_loss)
    self.actorcritic_optimizer.zero_grad()
    total_loss.backward()
    # torch.nn.utils.clip_grad_norm_(self.actorcritic.parameters(),500)
    self.actorcritic_optimizer.step()
    
#     print("*"*100)
#     print("Current Q:",curr_Q)
#     print("Next Q:",next_Q)
#     print("Estimated Q:",estimated_Q)
#     print("Value Loss:",critic_loss)
#     print("Entropy:",entropy)
#     print("Advantage",advantage)
#     print("Policy Loss:",policy_loss)
#     print("Total Loss:",total_loss)
#     print("*"*100)
    
    for name,param in self.actorcritic.named_parameters():
        if 'bn' not in name:
            self.writer.add_scalar(name,param.grad.norm(2).cpu().numpy(),episode)
    
    self.writer.add_scalar('Entropy loss',self.entropy_list[-1],len(self.entropy_list))
    self.writer.add_scalar('Value Loss',self.value_loss_list[-1],len(self.value_loss_list))
    self.writer.add_scalar('Policy Loss',self.policy_loss_list[-1],len(self.policy_loss_list))
    self.writer.add_scalar('Total Loss',self.total_loss_list[-1],len(self.total_loss_list))
