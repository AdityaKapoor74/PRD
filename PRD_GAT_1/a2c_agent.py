import numpy as np
import torch 
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
from torch.autograd import Variable
from torch.distributions import Categorical
from a2c import PolicyNetwork, CriticNetwork
import torch.nn.functional as F
import dgl
from torch.utils.data import DataLoader

class A2CAgent:

	def __init__(
		self, 
		env, 
		value_lr=2e-4, 
		policy_lr=2e-4, 
		entropy_pen=0.008, 
		gamma=0.99,
		lambda_ = 1e-3,
		trace_decay = 0.98,
		gif = False
		):

		self.env = env
		self.value_lr = value_lr
		self.policy_lr = policy_lr
		self.gamma = gamma
		self.entropy_pen = entropy_pen
		self.lambda_ = lambda_
		self.trace_decay = trace_decay
		self.tau = 0.999

		# self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		self.device = "cpu"
		
		self.num_agents = self.env.n
		self.num_actions = self.env.action_space[0].n
		self.gif = gif


		# self.critic_preprocess_input_dim = 2*3+2 # (pose,vel,goal pose, paired agent goal pose)
		self.critic_preprocess_input_dim = 2*3+2 #2*3+2
		self.critic_output_dim = 1
		# self.critic_network = CriticNetwork(self.critic_preprocess_input_dim, 16, self.critic_preprocess_input_dim, 16, 16+self.env.action_space[0].n, self.critic_output_dim, self.num_agents, self.env.action_space[0].n).to(self.device)
		self.critic_network = CriticNetwork(self.critic_preprocess_input_dim, 16, self.critic_preprocess_input_dim, 32, 16+self.env.action_space[0].n, self.critic_output_dim, self.num_agents, self.env.action_space[0].n).to(self.device)
		self.critic_network_target = CriticNetwork(self.critic_preprocess_input_dim, 16, self.critic_preprocess_input_dim, 32, 16+self.env.action_space[0].n, self.critic_output_dim, self.num_agents, self.env.action_space[0].n).to(self.device)
		self.critic_network_target.load_state_dict(self.critic_network.state_dict())


		self.policy_input_dim = 2*(3+2*(self.num_agents-1)) #2 for pose, 2 for vel and 2 for goal of current agent and rest (2 each) for relative position and relative velocity of other agents
		self.policy_output_dim = self.env.action_space[0].n
		policy_network_size = (self.policy_input_dim,512,256,self.policy_output_dim)
		self.policy_network = PolicyNetwork(policy_network_size).to(self.device)

		# weight assignment
		# self.weight_assignment = torch.zeros([self.num_agents,self.num_agents]).to(self.device)
		# for i in range(self.weight_assignment.shape[0]):
		# 	self.weight_assignment[i][self.num_agents-1-i] = 1

		# self.weight_loss = torch.nn.SmoothL1Loss(reduction="sum")

		# self.stop_policy_update = 100
		# self.update_both = 100
		# self.spu_counter = 0
		# self.ub_counter = 0



		# Loading models
		# model_path_value = "../../models/Experiment2/critic_networks/25-01-2021_VN_GAT1_PREPROC_GAT1_FC1_lr0.0002_PN_FC2_lr0.0002_GradNorm0.5_Entropy0.008_lambda0.1_epsiode46000.pt"
		# model_path_policy = "../../models/Experiment2/actor_networks/25-01-2021_PN_FC2_lr0.0002_VN_GAT1_PREPROC_GAT1_FC1_lr0.0002_GradNorm0.5_Entropy0.008_lambda0.1_epsiode46000.pt"
		# For CPU
		# self.critic_network.load_state_dict(torch.load(model_path_value,map_location=torch.device('cpu')))
		# self.policy_network.load_state_dict(torch.load(model_path_policy,map_location=torch.device('cpu')))
		# # For GPU
		# self.critic_network.load_state_dict(torch.load(model_path_value))
		# self.policy_network.load_state_dict(torch.load(model_path_policy))


		self.critic_optimizer = optim.Adam(self.critic_network.parameters(),lr=self.value_lr)
		self.policy_optimizer = optim.Adam(self.policy_network.parameters(),lr=self.policy_lr)


	def get_action(self,state):
		
		# dists = self.policy_network.forward(actor_graph)
		# actions = []
		# for i in range(self.num_agents):
		# 	probs = Categorical(dists[i])
		# 	index = probs.sample().cpu().detach().item()
		# 	actions.append(index)

		# return actions

		state = torch.FloatTensor(state).to(self.device)
		dists = self.policy_network.forward(state)
		probs = Categorical(dists)
		index = probs.sample().cpu().detach().item()

		return index



	def calculate_advantages(self,returns, values, rewards, dones, GAE = False, normalize = False):
		
		advantages = None

		if GAE:
			advantages = []
			next_value = 0
			advantage = 0
			rewards = rewards.unsqueeze(-1)
			dones = dones.unsqueeze(-1)
			masks = 1 - dones
			for t in reversed(range(0, len(rewards))):
				td_error = rewards[t] + (self.gamma * next_value * masks[t]) - values.data[t]
				next_value = values.data[t]
				
				advantage = td_error + (self.gamma * self.trace_decay * advantage * masks[t])
				advantages.insert(0, advantage)

			advantages = torch.stack(advantages)	
		else:
			advantages = returns - values
		
		if normalize:
			
			advantages = (advantages - advantages.mean()) / advantages.std()
		
		return advantages


	def calculate_returns(self,rewards, discount_factor, normalize = False):
	
		returns = []
		R = 0
		
		for r in reversed(rewards):
			R = r + R * discount_factor
			returns.insert(0, R)
		
		returns_tensor = torch.stack(returns).to(self.device)
		
		if normalize:
			
			returns_tensor = (returns_tensor - returns_tensor.mean()) / returns_tensor.std()
			
		return returns_tensor
		
		





	def update(self,critic_graphs,next_critic_graphs,one_hot_actions,one_hot_next_actions,actions,states_actor,next_states_actor,rewards,dones):

		'''
		Getting the probability mass function over the action space for each agent
		'''
		# probs = self.policy_network.forward(actor_graphs).reshape(-1,self.num_agents,self.num_actions)
		probs = self.policy_network.forward(states_actor)
		next_probs = self.policy_network.forward(next_states_actor)

		'''
		Calculate V values
		'''
		V_values, weights, weights_preproc = self.critic_network.forward(critic_graphs, probs.detach(), one_hot_actions)
		V_values_next, _, _ = self.critic_network_target.forward(next_critic_graphs, next_probs.detach(), one_hot_next_actions)
		V_values = V_values.reshape(-1,self.num_agents,self.num_agents)
		V_values_next = V_values.reshape(-1,self.num_agents,self.num_agents)
		weights = weights.reshape(-1,self.num_agents,self.num_agents)
		weights_preproc = weights_preproc.reshape(-1,self.num_agents,self.num_agents)

	# # ***********************************************************************************
	# 	#update critic (value_net)
	# we need a TxNxN vector so inflate the discounted rewards by N --> cloning the discounted rewards for an agent N times
		discounted_rewards = self.calculate_returns(rewards,self.gamma).unsqueeze(-2).repeat(1,self.num_agents,1)
		discounted_rewards = torch.transpose(discounted_rewards,-1,-2)
		target_values = torch.transpose(rewards.unsqueeze(-2).repeat(1,self.num_agents,1),-1,-2) + self.gamma*V_values_next*(1-dones.unsqueeze(-1))
		# value_loss = F.smooth_l1_loss(V_values,discounted_rewards) + self.lambda_*torch.sum(weights) #self.weight_loss(self.weight_assignment.unsqueeze(0).repeat(weights.shape[0],1,1),weights)#self.lambda_*F.smooth_l1_loss(self.weight_assignment.unsqueeze(0).repeat(weights.shape[0],1,1),weights)
		value_loss = F.smooth_l1_loss(V_values,target_values) + self.lambda_*torch.sum(weights)
		# # ***********************************************************************************
	# 	#update actor (policy net)
	# # ***********************************************************************************
		entropy = -torch.mean(torch.sum(probs * torch.log(torch.clamp(probs, 1e-10,1.0)), dim=2))

		# summing across each agent j to get the advantage
		# so we sum across the second last dimension which does A[t,j] = sum(V[t,i,j] - discounted_rewards[t,i])
		advantage = torch.sum(self.calculate_advantages(discounted_rewards, V_values, rewards, dones, True, False),dim=-2)

		probs = Categorical(probs)
		policy_loss = -probs.log_prob(actions) * advantage.detach()
		policy_loss = policy_loss.mean() - self.entropy_pen*entropy
	# # ***********************************************************************************
		
	# **********************************
		self.critic_optimizer.zero_grad()
		value_loss.backward(retain_graph=True)
		grad_norm_value = torch.nn.utils.clip_grad_norm_(self.critic_network.parameters(),0.5)
		self.critic_optimizer.step()


		self.policy_optimizer.zero_grad()
		policy_loss.backward(retain_graph=False)
		grad_norm_policy = torch.nn.utils.clip_grad_norm_(self.policy_network.parameters(),0.5)
		self.policy_optimizer.step()



		for target_param, param in zip(self.critic_network_target.parameters(), self.critic_network.parameters()):
			target_param.data.copy_(param.data * (1.0 - self.tau) + target_param.data * self.tau)
		
		# TRIN CRITIC > ACTOR
		# if self.update_both > self.ub_counter:
		# 	self.critic_optimizer.zero_grad()
		# 	value_loss.backward(retain_graph=True)
		# 	grad_norm_value = torch.nn.utils.clip_grad_norm_(self.critic_network.parameters(),0.5)
		# 	self.critic_optimizer.step()


		# 	self.policy_optimizer.zero_grad()
		# 	policy_loss.backward(retain_graph=False)
		# 	grad_norm_policy = torch.nn.utils.clip_grad_norm_(self.policy_network.parameters(),0.5)
		# 	self.policy_optimizer.step()

		# 	self.ub_counter += 1
		# 	self.spu_counter = 0
		# else:
		# 	self.critic_optimizer.zero_grad()
		# 	value_loss.backward(retain_graph=False)
		# 	grad_norm_value = torch.nn.utils.clip_grad_norm_(self.critic_network.parameters(),0.5)
		# 	self.critic_optimizer.step()

		# 	grad_norm_policy = -100

		# 	if self.spu_counter > self.stop_policy_update:
		# 		self.ub_counter = 0

		# 	self.spu_counter += 1
	# # ***********************************************************************************
		return value_loss,policy_loss,entropy,grad_norm_value,grad_norm_policy,weights,weights_preproc