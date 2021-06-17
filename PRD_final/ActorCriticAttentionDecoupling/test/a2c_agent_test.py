import numpy as np
import torch 
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
from torch.autograd import Variable
from torch.distributions import Categorical
from a2c_test import StateActionGATCritic, StateOnlyGATCritic, StateOnlyMLPCritic, StateActionMLPCritic, MLPPolicyNetwork
import torch.nn.functional as F

class A2CAgent:

	def __init__(
		self, 
		env, 
		dictionary
		):

		self.env = env
		self.value_lr = dictionary["value_lr"]
		self.policy_lr = dictionary["policy_lr"]
		self.gamma = dictionary["gamma"]
		self.entropy_pen = dictionary["entropy_pen"]
		self.trace_decay = dictionary["trace_decay"]
		self.top_k = dictionary["top_k"]
		self.critic_type = dictionary["critic_type"]
		self.gae = dictionary["gae"]
		self.norm_adv = dictionary["norm_adv"]
		self.norm_rew = dictionary["norm_rew"]
		# Used for masking advantages above a threshold
		self.select_above_threshold = dictionary["select_above_threshold"]
		# cut the tail of softmax --> Used in softmax with normalization
		self.softmax_cut_threshold = dictionary["softmax_cut_threshold"]

		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		# self.device = "cpu"
		
		self.num_agents = self.env.n
		self.num_actions = self.env.action_space[0].n
		self.gif = dictionary["gif"]

		self.experiment_type = dictionary["experiment_type"]
		self.scaling_factor = None
		if self.experiment_type == "without_prd_scaled" or self.experiment_type == "with_prd_soft_adv_scaled":
			self.scaling_factor = self.num_agents
		elif "top" in self.experiment_type:
			self.scaling_factor = self.num_agents/self.top_k

		self.greedy_policy = torch.zeros(self.num_agents,self.num_agents).to(self.device)
		for i in range(self.num_agents):
			self.greedy_policy[i][i] = 1

		print("CRITIC TYPE", self.critic_type)
		print("EXPERIMENT TYPE", self.experiment_type)

		# TD lambda
		self.lambda_ = 0.8

		# MLP_CRITIC_STATE, MLP_CRITIC_STATE_ACTION, GNN_CRITIC_STATE, GNN_CRITIC_STATE_ACTION
		if self.critic_type == "MLP_CRITIC_STATE":
			self.critic_network = StateOnlyMLPCritic(2*4, self.num_agents).to(self.device)
		elif self.critic_type == "MLP_CRITIC_STATE_ACTION":
			self.critic_network = StateActionMLPCritic(2*4, self.num_actions, self.num_agents).to(self.device)
		elif self.critic_type == "GNN_CRITIC_STATE":
			self.critic_network = StateOnlyGATCritic(2*4, 128, 128, 1, self.num_agents, self.num_actions).to(self.device)
		elif self.critic_type == "GNN_CRITIC_STATE_ACTION":
			self.critic_network = StateActionGATCritic(2*4, 128, 2*4+self.num_actions, 128, 128, 1, self.num_agents, self.num_actions).to(self.device)
		elif self.critic_type == "ALL":
			self.critic_network_1 = StateOnlyMLPCritic(2*4, self.num_agents).to(self.device)
			self.critic_network_2 = StateActionMLPCritic(2*4, self.num_actions, self.num_agents).to(self.device)
			self.critic_network_3 = StateOnlyGATCritic(2*4, 128, 128, 1, self.num_agents, self.num_actions).to(self.device)
			self.critic_network_4 = StateActionGATCritic(2*4, 128, 2*4+self.num_actions, 128, 128, 1, self.num_agents, self.num_actions).to(self.device)

		# MLP POLICY
		self.policy_network = MLPPolicyNetwork(2*3, self.num_agents, self.num_actions).to(self.device)


		# Loading models
		# model_path_value = "../../../models/Experiment37/critic_networks/11-04-2021VN_SAT_SAT_FCN_lr0.0002_PN_FCN_lr0.0002_GradNorm0.5_Entropy0.008_trace_decay0.98_lambda0.0tanh_epsiode6000.pt"
		model_path_policy = "./test2/16-06-2021_PN_ATN_FCN_lr0.001VN_SAT_FCN_lr0.01_GradNorm0.5_Entropy0.008_trace_decay0.98topK_0select_above_threshold0.1softmax_cut_threshold0.1_epsiode15000.pt"
		# For CPU
		# self.critic_network.load_state_dict(torch.load(model_path_value,map_location=torch.device('cpu')))
		# self.policy_network.load_state_dict(torch.load(model_path_policy,map_location=torch.device('cpu')))
		# # For GPU
		# self.critic_network.load_state_dict(torch.load(model_path_value))
		self.policy_network.load_state_dict(torch.load(model_path_policy))

		if self.critic_type == "ALL":
			self.critic_optimizer_1 = optim.Adam(self.critic_network_1.parameters(),lr=self.value_lr)
			self.critic_optimizer_2 = optim.Adam(self.critic_network_2.parameters(),lr=self.value_lr)
			self.critic_optimizer_3 = optim.Adam(self.critic_network_3.parameters(),lr=self.value_lr)
			self.critic_optimizer_4 = optim.Adam(self.critic_network_4.parameters(),lr=self.value_lr)
		else:
			self.critic_optimizer = optim.Adam(self.critic_network.parameters(),lr=self.value_lr)
			self.policy_optimizer = optim.Adam(self.policy_network.parameters(),lr=self.policy_lr)


	def get_scaling_factor(self):
		if self.experiment_type == "without_prd_scaled" or self.experiment_type == "with_prd_soft_adv_scaled":
			self.scaling_factor = self.num_agents
		elif "top" in self.experiment_type:
			self.scaling_factor = self.num_agents/self.top_k

		return self.scaling_factor

	def get_action(self,state):
		state = torch.FloatTensor([state]).to(self.device)
		dists, _ = self.policy_network.forward(state)
		index = [Categorical(dist).sample().cpu().detach().item() for dist in dists[0]]
		return index



	def calculate_advantages(self,returns, values, rewards, dones):
		
		advantages = None

		if self.gae:
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
		
		if self.norm_adv:
			
			advantages = (advantages - advantages.mean()) / advantages.std()
		
		return advantages


	def calculate_deltas(self, values, rewards, dones):
		deltas = []
		next_value = 0
		rewards = rewards.unsqueeze(-1)
		dones = dones.unsqueeze(-1)
		masks = 1-dones
		for t in reversed(range(0, len(rewards))):
			td_error = rewards[t] + (self.gamma * next_value * masks[t]) - values.data[t]
			next_value = values.data[t]
			deltas.insert(0,td_error)
		deltas = torch.stack(deltas)

		return deltas


	def nstep_returns(self,values, rewards, dones):
		deltas = self.calculate_deltas(values, rewards, dones)
		advs = self.calculate_returns(deltas, self.gamma*self.lambda_)
		target_Vs = advs+values
		return target_Vs


	def calculate_returns(self,rewards, discount_factor):
		returns = []
		R = 0
		
		for r in reversed(rewards):
			R = r + R * discount_factor
			returns.insert(0, R)
		
		returns_tensor = torch.stack(returns).to(self.device)
		
		if self.norm_rew:
			
			returns_tensor = (returns_tensor - returns_tensor.mean()) / returns_tensor.std()
			
		return returns_tensor
		
		


	def update(self,states_critic,next_states_critic,one_hot_actions,one_hot_next_actions,actions,states_actor,next_states_actor,rewards,dones):

		'''
		Getting the probability mass function over the action space for each agent
		'''
		probs, weight_policy = self.policy_network.forward(states_actor)

		'''
		Calculate V values
		'''
		if self.critic_type == "ALL":
			value_loss = []
			grad_norm_value = []
			weights = []
			for i in range(1,5):
				if i == 1:
					V_values, weights_ = self.critic_network_1.forward(states_critic, probs.detach(), one_hot_actions)
					# V_values_next, _ = self.critic_network_1.forward(next_states_critic, next_probs.detach(), one_hot_next_actions)
				elif i == 2:
					V_values, weights_ = self.critic_network_2.forward(states_critic, probs.detach(), one_hot_actions)
					# V_values_next, _ = self.critic_network_2.forward(next_states_critic, next_probs.detach(), one_hot_next_actions)
				elif i == 3:
					V_values, weights_ = self.critic_network_3.forward(states_critic, probs.detach(), one_hot_actions)
					# V_values_next, _ = self.critic_network_3.forward(next_states_critic, next_probs.detach(), one_hot_next_actions)
				elif i == 4:
					V_values, weights_ = self.critic_network_4.forward(states_critic, probs.detach(), one_hot_actions)
					# V_values_next, _ = self.critic_network_4.forward(next_states_critic, next_probs.detach(), one_hot_next_actions)
				


				V_values = V_values.reshape(-1,self.num_agents,self.num_agents)
				# V_values_next = V_values_next.reshape(-1,self.num_agents,self.num_agents)

			
				# # ***********************************************************************************
				# update critic (value_net)
				# we need a TxNxN vector so inflate the discounted rewards by N --> cloning the discounted rewards for an agent N times
				# discounted_rewards = self.calculate_returns(rewards,self.gamma).unsqueeze(-2).repeat(1,self.num_agents,1).to(self.device)
				# discounted_rewards = torch.transpose(discounted_rewards,-1,-2)

				# BOOTSTRAP LOSS
				# target_values = torch.transpose(rewards.unsqueeze(-2).repeat(1,self.num_agents,1),-1,-2) + self.gamma*V_values_next*(1-dones.unsqueeze(-1))
				# value_loss = F.smooth_l1_loss(V_values,target_values)

				# MONTE CARLO LOSS
				# value_loss = F.smooth_l1_loss(V_values,discounted_rewards)

				# TD lambda 
				Value_target_ = self.nstep_returns(V_values, rewards, dones)
				value_loss_ = F.smooth_l1_loss(V_values, Value_target_)

				if i == 1:
					self.critic_optimizer_1.zero_grad()
					value_loss_.backward(retain_graph=False)
					grad_norm_value_ = torch.nn.utils.clip_grad_norm_(self.critic_network_1.parameters(),100.0)
					self.critic_optimizer_1.step()
				elif i == 2:
					self.critic_optimizer_2.zero_grad()
					value_loss_.backward(retain_graph=False)
					grad_norm_value_ = torch.nn.utils.clip_grad_norm_(self.critic_network_2.parameters(),100.0)
					self.critic_optimizer_2.step()
				elif i == 3:
					self.critic_optimizer_3.zero_grad()
					value_loss_.backward(retain_graph=False)
					grad_norm_value_ = torch.nn.utils.clip_grad_norm_(self.critic_network_3.parameters(),100.0)
					self.critic_optimizer_3.step()
				elif i == 4:
					self.critic_optimizer_4.zero_grad()
					value_loss_.backward(retain_graph=False)
					grad_norm_value_ = torch.nn.utils.clip_grad_norm_(self.critic_network_4.parameters(),100.0)
					self.critic_optimizer_4.step()

				value_loss.append(value_loss_)
				grad_norm_value.append(grad_norm_value_)
				weights.append(weights_)

			return value_loss,None,None,grad_norm_value,None,weights,None

		else:
			V_values, weights = self.critic_network.forward(states_critic, probs.detach(), one_hot_actions)
			# V_values_next, _ = self.critic_network.forward(next_states_critic, next_probs.detach(), one_hot_next_actions)
			V_values = V_values.reshape(-1,self.num_agents,self.num_agents)
			# V_values_next = V_values_next.reshape(-1,self.num_agents,self.num_agents)

		
			# # ***********************************************************************************
			# update critic (value_net)
			# we need a TxNxN vector so inflate the discounted rewards by N --> cloning the discounted rewards for an agent N times
			discounted_rewards = self.calculate_returns(rewards,self.gamma).unsqueeze(-2).repeat(1,self.num_agents,1).to(self.device)
			discounted_rewards = torch.transpose(discounted_rewards,-1,-2)

			# BOOTSTRAP LOSS
			# target_values = torch.transpose(rewards.unsqueeze(-2).repeat(1,self.num_agents,1),-1,-2) + self.gamma*V_values_next*(1-dones.unsqueeze(-1))
			# value_loss = F.smooth_l1_loss(V_values,target_values)

			# MONTE CARLO LOSS
			# value_loss = F.smooth_l1_loss(V_values,discounted_rewards)

			# TD lambda 
			Value_target = self.nstep_returns(V_values, rewards, dones)
			value_loss = F.smooth_l1_loss(V_values, discounted_rewards)
		
			# # ***********************************************************************************
			# update actor (policy net)
			# # ***********************************************************************************
			entropy = -torch.mean(torch.sum(probs * torch.log(torch.clamp(probs, 1e-10,1.0)), dim=2))

			# summing across each agent j to get the advantage
			# so we sum across the second last dimension which does A[t,j] = sum(V[t,i,j] - discounted_rewards[t,i])
			advantage = None
			if self.experiment_type == "without_prd" or self.experiment_type == "without_prd_scaled":
				advantage = torch.sum(self.calculate_advantages(discounted_rewards, V_values, rewards, dones),dim=-2)
			elif "top" in self.experiment_type:
				values, indices = torch.topk(weights,k=self.top_k,dim=-1)
				masking_advantage = torch.transpose(torch.sum(F.one_hot(indices, num_classes=self.num_agents), dim=-2),-1,-2)
				advantage = torch.sum(self.calculate_advantages(discounted_rewards, V_values, rewards, dones) * masking_advantage,dim=-2)
			elif self.experiment_type in "above_threshold":
				masking_advantage = torch.transpose((weights>self.select_above_threshold).int(),-1,-2)
				advantage = torch.sum(self.calculate_advantages(discounted_rewards, V_values, rewards, dones) * masking_advantage,dim=-2)
			elif self.experiment_type == "with_prd_soft_adv" or self.experiment_type == "with_prd_soft_adv_scaled":
				advantage = torch.sum(self.calculate_advantages(discounted_rewards, V_values, rewards, dones) * weights.transpose(-1,-2) ,dim=-2)
			elif self.experiment_type == "greedy_policy":
				advantage = torch.sum(self.calculate_advantages(discounted_rewards, V_values, rewards, dones) * self.greedy_policy ,dim=-2)


			if "scaled" in self.experiment_type:
				advantage = advantage * self.scaling_factor
		
			probs = Categorical(probs)
			policy_loss = -probs.log_prob(actions) * advantage.detach()
			policy_loss = policy_loss.mean() - self.entropy_pen*entropy
			# # ***********************************************************************************
				
			# **********************************
			self.critic_optimizer.zero_grad()
			value_loss.backward(retain_graph=False)
			grad_norm_value = torch.nn.utils.clip_grad_norm_(self.critic_network.parameters(),0.5)
			self.critic_optimizer.step()


			self.policy_optimizer.zero_grad()
			policy_loss.backward(retain_graph=False)
			grad_norm_policy = torch.nn.utils.clip_grad_norm_(self.policy_network.parameters(),0.5)
			self.policy_optimizer.step()

			# V values
			return value_loss,policy_loss,entropy,grad_norm_value,grad_norm_policy,weights,weight_policy