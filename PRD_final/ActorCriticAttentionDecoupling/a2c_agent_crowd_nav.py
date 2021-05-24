import numpy as np
import torch 
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
from torch.autograd import Variable
from torch.distributions import Categorical
from a2c_crowd_nav import ScalarDotProductCriticNetwork, ScalarDotProductPolicyNetwork
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
		# Used for masking advantages above a threshold
		self.select_above_threshold = dictionary["select_above_threshold"]
		# cut the tail of softmax --> Used in softmax with normalization
		self.softmax_cut_threshold = dictionary["softmax_cut_threshold"]

		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		# self.device = "cpu"
		
		self.num_agents = dictionary["num_agents"]
		self.num_people = dictionary["num_people"]
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


		print(self.experiment_type, self.scaling_factor)


		# SCALAR DOT PRODUCT
		self.obs_agent_input_dim = 2*3
		self.obs_act_agent_input_dim = self.obs_agent_input_dim + self.num_actions # (pose,vel,goal pose, paired agent goal pose) --> observations 
		self.obs_act_agent_output_dim = 16
		self.obs_people_input_dim = 2*2
		self.obs_act_people_input_dim = self.obs_people_input_dim + self.num_actions # (pose,vel,goal pose, paired agent goal pose) --> observations 
		self.obs_act_people_output_dim = 16
		self.final_input_dim = self.obs_act_agent_output_dim + self.obs_act_people_output_dim
		self.final_output_dim = 1
		self.critic_network = ScalarDotProductCriticNetwork(self.obs_act_agent_input_dim, self.obs_act_agent_output_dim, self.obs_act_people_input_dim, self.obs_act_people_output_dim, self.final_input_dim, self.final_output_dim, self.num_agents, self.num_people, self.num_actions, self.softmax_cut_threshold).to(self.device)
		
		
		# SCALAR DOT PRODUCT POLICY NETWORK
		self.obs_agent_input_dim = 2*3
		self.obs_agent_output_dim = 16
		self.obs_people_input_dim = 2*2 # (pose,vel,goal pose, paired agent goal pose) --> observations 
		self.obs_people_output_dim = 16
		self.final_input_dim = self.obs_agent_output_dim + self.obs_people_output_dim
		self.final_output_dim = self.num_actions
		self.policy_network = ScalarDotProductPolicyNetwork(self.obs_agent_input_dim, self.obs_agent_output_dim, self.obs_people_input_dim, self.obs_people_output_dim, self.final_input_dim, self.final_output_dim, self.num_agents, self.num_people, self.num_actions, self.softmax_cut_threshold).to(self.device)


		# Loading models
		# model_path_value = "../../../models/Scalar_dot_product/crowd_nav/6_Agents_4_People/SingleAttentionMechanism/without_prd/critic_networks/24-05-2021VN_ATN_FCN_lr0.01_PN_ATN_FCN_lr0.001_GradNorm0.5_Entropy0.008_trace_decay0.98topK_2select_above_threshold0.1softmax_cut_threshold0.1_epsiode80000.pt"
		# model_path_policy = "../../../models/Scalar_dot_product/crowd_nav/6_Agents_4_People/SingleAttentionMechanism/without_prd/actor_networks/24-05-2021_PN_ATN_FCN_lr0.001VN_SAT_FCN_lr0.01_GradNorm0.5_Entropy0.008_trace_decay0.98topK_2select_above_threshold0.1softmax_cut_threshold0.1_epsiode80000.pt"
		# For CPU
		# self.critic_network.load_state_dict(torch.load(model_path_value,map_location=torch.device('cpu')))
		# self.policy_network.load_state_dict(torch.load(model_path_policy,map_location=torch.device('cpu')))
		# # For GPU
		# self.critic_network.load_state_dict(torch.load(model_path_value))
		# self.policy_network.load_state_dict(torch.load(model_path_policy))


		self.critic_optimizer = optim.Adam(self.critic_network.parameters(),lr=self.value_lr)
		self.policy_optimizer = optim.Adam(self.policy_network.parameters(),lr=self.policy_lr)


	def get_action(self,state, state_people):
		# MLP
		# state = torch.FloatTensor(state).to(self.device)
		# dists = self.policy_network.forward(state)
		# probs = Categorical(dists)
		# index = probs.sample().cpu().detach().item()

		# return index

		# GNN
		state = torch.FloatTensor([state]).to(self.device)
		state_people = torch.FloatTensor([state_people]).to(self.device)
		dists, _, _ = self.policy_network.forward(state, state_people)
		index = [Categorical(dist).sample().cpu().detach().item() for dist in dists[0]]
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
		
		





	def update(self,states_critic,next_states_critic,one_hot_actions,one_hot_next_actions,actions,states_actor,next_states_actor,rewards,dones,states_critic_people,next_states_critic_people,one_hot_actions_people,one_hot_next_actions_people,states_actor_people,next_states_actor_people):

		'''
		Getting the probability mass function over the action space for each agent
		'''
		# MLP
		# probs = self.policy_network.forward(actor_graphs).reshape(-1,self.num_agents,self.num_actions)
		# probs = self.policy_network.forward(states_actor)
		# next_probs = self.policy_network.forward(next_states_actor)

		# GNN
		probs, weight_policy, weight_policy_people = self.policy_network.forward(states_actor,states_actor_people)

		'''
		Calculate V values
		'''
		V_values, weights, weights_people = self.critic_network.forward(states_critic, probs.detach(), one_hot_actions, states_critic_people, one_hot_actions_people)
		# V_values_next, _ = self.critic_network.forward(next_states_critic, next_probs.detach(), one_hot_next_actions)
		V_values = V_values.reshape(-1,self.num_agents,self.num_agents)
		# V_values_next = V_values_next.reshape(-1,self.num_agents,self.num_agents)

		
	# # ***********************************************************************************
	# 	#update critic (value_net)
	# we need a TxNxN vector so inflate the discounted rewards by N --> cloning the discounted rewards for an agent N times
		discounted_rewards = self.calculate_returns(rewards,self.gamma).unsqueeze(-2).repeat(1,self.num_agents,1).to(self.device)
		discounted_rewards = torch.transpose(discounted_rewards,-1,-2)

		# BOOTSTRAP LOSS
		# target_values = torch.transpose(rewards.unsqueeze(-2).repeat(1,self.num_agents,1),-1,-2) + self.gamma*V_values_next*(1-dones.unsqueeze(-1))
		# value_loss = F.smooth_l1_loss(V_values,target_values)

		# MONTE CARLO LOSS
		value_loss = F.smooth_l1_loss(V_values,discounted_rewards)

		# REGRESSING ON IMMEDIATE REWARD
		# value_loss = F.smooth_l1_loss(V_values,torch.transpose(rewards.unsqueeze(-2).repeat(1,self.num_agents,1),-1,-2))
		
		# # ***********************************************************************************
	# 	#update actor (policy net)
	# # ***********************************************************************************
		entropy = -torch.mean(torch.sum(probs * torch.log(torch.clamp(probs, 1e-10,1.0)), dim=2))

		# summing across each agent j to get the advantage
		# so we sum across the second last dimension which does A[t,j] = sum(V[t,i,j] - discounted_rewards[t,i])
		advantage = None
		if self.experiment_type == "without_prd" or self.experiment_type == "without_prd_scaled":
			advantage = torch.sum(self.calculate_advantages(discounted_rewards, V_values, rewards, dones, True, False),dim=-2)
		elif "top" in self.experiment_type:
			values, indices = torch.topk(weights,k=self.top_k,dim=-1)
			masking_advantage = torch.transpose(torch.sum(F.one_hot(indices, num_classes=self.num_agents), dim=-2),-1,-2)
			advantage = torch.sum(self.calculate_advantages(discounted_rewards, V_values, rewards, dones, True, False) * masking_advantage,dim=-2)
		elif self.experiment_type in "above_threshold":
			masking_advantage = torch.transpose((weights>self.select_above_threshold).int(),-1,-2)
			advantage = torch.sum(self.calculate_advantages(discounted_rewards, V_values, rewards, dones, True, False) * masking_advantage,dim=-2)
		elif self.experiment_type == "with_prd_soft_adv" or self.experiment_type == "with_prd_soft_adv_scaled":
			advantage = torch.sum(self.calculate_advantages(discounted_rewards, V_values, rewards, dones, True, False) * weights.transpose(-1,-2) ,dim=-2)
		elif self.experiment_type == "greedy_policy":
			advantage = torch.sum(self.calculate_advantages(discounted_rewards, V_values, rewards, dones, True, False) * self.greedy_policy ,dim=-2)

		if "scaled" in self.experiment_type:
			advantage = advantage * self.scaling_factor


		probs = Categorical(probs)
		policy_loss = -probs.log_prob(actions) * advantage.detach()
		policy_loss = policy_loss.mean() - self.entropy_pen*entropy
		# policy_loss = torch.Tensor([0])
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
		# grad_norm_policy = 0

		# V values
		return value_loss,policy_loss,entropy,grad_norm_value,grad_norm_policy,weights, weight_policy, weights_people, weight_policy_people