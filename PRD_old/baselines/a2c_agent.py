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
		gif=False
		):

		self.env = env
		self.value_lr = value_lr
		self.policy_lr = policy_lr
		self.gamma = gamma
		self.entropy_pen = entropy_pen

		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		
		self.num_agents = self.env.n

		self.gif = gif


		self.value_input_dim = 2*3 + (2*3+self.env.action_space[0].n)*(self.num_agents-1) # for pose,vel and landmark of current agent followed by pose, vel, landmark and one-hot actions
		self.value_output_dim = 1 # State-Value
		self.value_network = ValueNetwork(self.value_input_dim, self.value_output_dim).to(self.device)
		
		self.policy_input_dim = 2*(3+2*(self.num_agents-1)) #2 for pose, 2 for vel and 2 for goal of current agent and rest (2 each) for relative position and relative velocity of other agents
		self.policy_output_dim = self.env.action_space[0].n
		self.policy_network = PolicyNetwork(self.policy_input_dim, self.policy_output_dim).to(self.device)


		self.value_optimizer = optim.Adam(self.value_network.parameters(),lr=self.value_lr)
		self.policy_optimizer = optim.Adam(self.policy_network.parameters(),lr=self.policy_lr)

		if self.gif:
			# Loading models

			model_path_value = "../../models/baselines/4_agents/value_net_2_layered_lr_2e-4_policy_lr_2e-4_with_grad_norm_0.5_entropy_pen_0.008_xavier_uniform_init_policy.pt"
			model_path_policy = "../../models/baselines/4_agents/policy_net_lr_2e-4_value_2_layered_lr_2e-4_with_grad_norm_0.5_entropy_pen_0.008_xavier_uniform_init_policy.pt"
			# For CPU
			# self.value_network.load_state_dict(torch.load(model_path_value,map_location=torch.device('cpu')))
			# self.policy_network.load_state_dict(torch.load(model_path_policy,map_location=torch.device('cpu')))
			# For GPU
			self.value_network.load_state_dict(torch.load(model_path_value))
			self.policy_network.load_state_dict(torch.load(model_path_policy))


		
	

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



	def update(self,states_critic,states_actor,actions,rewards,dones):

		'''
		Getting the probability mass function over the action space for each agent
		'''
		probs = self.policy_network.forward(states_actor)

		'''
		Calculate V values
		'''
		V_values = self.value_network.forward(states_critic).reshape(-1,self.num_agents)



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

		# value_loss = F.smooth_l1_loss(V_values,discounted_rewards)


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