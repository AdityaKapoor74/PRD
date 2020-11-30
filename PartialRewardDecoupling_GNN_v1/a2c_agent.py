import numpy as np
import torch 
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
from torch.autograd import Variable
from torch.distributions import Categorical
from a2c import PolicyNetwork, ValueNetwork
import torch.nn.functional as F
from fraction import Fraction

class A2CAgent:

	def __init__(
		self, 
		env, 
		value_lr=2e-4, 
		policy_lr=2e-4, 
		entropy_pen=0.008, 
		gamma=0.99,
		gif = False
		):

		self.env = env
		self.value_lr = value_lr
		self.policy_lr = policy_lr
		self.gamma = gamma
		self.entropy_pen = entropy_pen

		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		
		self.num_agents = self.env.n
		self.gif = gif


		self.value_input_dim_curr_agent = 2*3 # for pose,vel and landmark
		self.value_input_dim_other_agent = 2*3 + self.env.action_space[0].n # for pose,vel and landmark along with one hot actions for one agent
		self.value_output_dim = 1 # State-Value
		current_agent_size = (self.value_input_dim_curr_agent,128)
		other_agent_size = (self.value_input_dim_other_agent,128)
		common_size = (128,1)
		self.value_network = ValueNetwork(current_agent_size,other_agent_size,common_size,self.num_agents).to(self.device)
		
		self.policy_input_dim = 2*(3+2*(self.num_agents-1)) #2 for pose, 2 for vel and 2 for goal of current agent and rest (2 each) for relative position and relative velocity of other agents
		self.policy_output_dim = self.env.action_space[0].n
		policy_network_size = (self.policy_input_dim,512,256,self.policy_output_dim)
		self.policy_network = PolicyNetwork(policy_network_size).to(self.device)


		self.value_optimizer = optim.Adam(self.value_network.parameters(),lr=self.value_lr)
		self.policy_optimizer = optim.Adam(self.policy_network.parameters(),lr=self.policy_lr)



		
	

	def get_action(self,state):
		state = torch.FloatTensor(state).to(self.device)
		dists = self.policy_network.forward(state)
		probs = Categorical(dists)
		index = probs.sample().cpu().detach().item()

		return index



	def calculate_advantages(self,returns, values, normalize = False):
	
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


	def get_one_hot_encoding(self,actions):
		one_hot = torch.zeros([actions.shape[0], self.num_agents, self.env.action_space[0].n], dtype=torch.int32)
		for i in range(one_hot.shape[0]):
			for j in range(self.num_agents):
				one_hot[i][j][int(actions[i][j].item())] = 1

		return one_hot



	def update(self,current_agent_critic,other_agent_critic,states_actor,actions,rewards,dones):

		'''
		Getting the probability mass function over the action space for each agent
		'''
		probs = self.policy_network.forward(states_actor)

		'''
		Calculate V values
		'''
		V_values = self.value_network.forward(current_agent_critic.unsqueeze(-2),other_agent_critic).reshape(-1,self.num_agents)



	# # ***********************************************************************************
	# 	#update critic (value_net)
		discounted_rewards = self.calculate_returns(rewards,self.gamma)

		value_loss = F.smooth_l1_loss(V_values,discounted_rewards)

	# # ***********************************************************************************
	# 	#update actor (policy net)
	# # ***********************************************************************************

		entropy = -torch.mean(torch.sum(probs * torch.log(torch.clamp(probs, 1e-10,1.0)), dim=2))

		# summing across each agent 
		value_targets = torch.sum(discounted_rewards,dim=1) 
		value_estimates = torch.sum(V_values,dim=1)


		advantage = self.calculate_advantages(value_targets, value_estimates)
		probs = Categorical(probs)
		policy_loss = -probs.log_prob(actions) * advantage.unsqueeze(-1).detach()
		policy_loss = policy_loss.mean() - self.entropy_pen*entropy
	# # ***********************************************************************************
		
	# # *************************************************
	# **********************************
		self.value_optimizer.zero_grad()
		value_loss.backward(retain_graph=False)
		grad_norm_value = torch.nn.utils.clip_grad_norm_(self.value_network.parameters(),0.5)
		self.value_optimizer.step()
		

		self.policy_optimizer.zero_grad()
		policy_loss.backward(retain_graph=False)
		grad_norm_policy = torch.nn.utils.clip_grad_norm_(self.policy_network.parameters(),0.5)
		self.policy_optimizer.step()
	# # ***********************************************************************************
		return value_loss,policy_loss,entropy,grad_norm_value,grad_norm_policy