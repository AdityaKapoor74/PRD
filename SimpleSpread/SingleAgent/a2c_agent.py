import numpy as np
import torch 
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
from torch.autograd import Variable
from torch.distributions import Categorical
from a2c import *
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
import gc

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
    model_path = "/home/aditya/Desktop/Partial_Reward_Decoupling/PRD/SimpleSpread/models/actorcritic_network_lr_2e-4_with_grad_norm_1_entropy_pen_0.008_xavier_init_clamp_logs.pt"
    self.actorcritic.load_state_dict(torch.load(model_path,map_location=torch.device('cpu')))

    self.MSELoss = nn.MSELoss()
    self.actorcritic_optimizer = optim.Adam(self.actorcritic.parameters(),lr=lr)

    self.entropy_list = []
    self.value_loss_list = []
    self.policy_loss_list = []
    self.total_loss_list = []
    self.writer = SummaryWriter('runs/simple_spread_lr_2e-4_with_grad_norm_1_entropy_pen_0.008_xavier_init_clamp_logs')

  def get_action(self,state):
    state = torch.FloatTensor(state).to(self.device)
    logits,_ = self.actorcritic.forward(state)
    # print("logits:",logits)
    del state
    dist = F.softmax(logits,dim=0)
    # print("dist:",dist)
    del logits
    # print('dist: ', dist)
    probs = Categorical(dist)
    # print("PROBS:",probs)
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
    # print("STATE BATCH:")
    # print(global_state_batch.shape)
    curr_logits,curr_Q = self.actorcritic.forward(global_state_batch)
    # print("CURRENT LOGITS:")
    # print(curr_logits.shape)
    rewards = rewards.reshape(-1,1)
    _,next_Q = self.actorcritic.forward(global_next_state_batch)
    estimated_Q = rewards + self.gamma*next_Q
    

    # critic_loss = self.MSELoss(curr_Q,estimated_Q.detach())
    critic_loss = F.smooth_l1_loss(curr_Q,estimated_Q.detach())
    dists = F.softmax(curr_logits,dim=1)
    probs = Categorical(dists)

    entropy = []
    for dist in dists:
      entropy.append(-torch.sum(dist*torch.log(torch.clamp(dist,1e-10,1))))
    entropy = torch.stack(entropy).mean()
    self.entropy_list.append(entropy.item())

    # print('self.entropy_list: ', self.entropy_list)

    advantage = estimated_Q - curr_Q
    policy_loss = -probs.log_prob(global_actions_batch.view(global_actions_batch.size(0))).view(-1, 1) * advantage.detach()
    policy_loss = policy_loss.mean()
    self.policy_loss_list.append(policy_loss.item())
    self.value_loss_list.append(critic_loss.item())
    total_loss = policy_loss + critic_loss - 0.008*entropy
    self.total_loss_list.append(total_loss.item())
    self.actorcritic_optimizer.zero_grad()
    total_loss.backward(retain_graph=False)
    grad_norm = torch.nn.utils.clip_grad_norm_(self.actorcritic.parameters(),1)
    self.actorcritic_optimizer.step()
    # total_loss.detach()
    
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
            self.writer.add_scalar('Gradients/'+str(name),param.grad.norm(2).cpu().numpy(),episode)
    
    self.writer.add_scalar('Loss/Entropy loss',self.entropy_list[-1],len(self.entropy_list))
    self.writer.add_scalar('Loss/Value Loss',self.value_loss_list[-1],len(self.value_loss_list))
    self.writer.add_scalar('Loss/Policy Loss',self.policy_loss_list[-1],len(self.policy_loss_list))
    self.writer.add_scalar('Loss/Total Loss',self.total_loss_list[-1],len(self.total_loss_list))
    self.writer.add_scalar('Gradient Normalization/Grad Norm',grad_norm,episode)


    # for obj in gc.get_objects():
    #     try:
    #         if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
    #             print(type(obj), obj.size())
    #     except: pass
