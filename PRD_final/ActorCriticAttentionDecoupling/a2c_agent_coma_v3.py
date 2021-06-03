import numpy as np
import torch 
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
from torch.autograd import Variable
from torch.distributions import Categorical
from a2c_coma_v3 import ScalarDotProductCriticNetwork_Q, ScalarDotProductCriticNetwork_V, ScalarDotProductPolicyNetwork
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

		print(self.experiment_type, self.scaling_factor)


		# SCALAR DOT PRODUCT FOR Q NETWORK
		# self.obs_input_dim = 2*4
		# self.obs_output_dim = 16
		# self.obs_act_input_dim = self.obs_input_dim + self.num_actions # (pose,vel,goal pose, paired agent goal pose) --> observations 
		# self.obs_act_output_dim = 16
		# self.final_input_dim = self.obs_act_output_dim #+ self.obs_input_dim #self.obs_z_output_dim + self.weight_input_dim
		# self.final_output_dim = self.num_actions
		# self.critic_network_Q = ScalarDotProductCriticNetwork_Q(self.obs_input_dim, self.obs_output_dim, self.obs_act_input_dim, self.obs_act_output_dim, self.final_input_dim, self.final_output_dim, self.num_agents, self.num_actions, self.softmax_cut_threshold).to(self.device)
		
		# SCALAR DOT PRODUCT FOR V NETWORK
		self.obs_input_dim = 2*4
		self.obs_output_dim = 16
		self.obs_act_input_dim = self.obs_input_dim + self.num_actions # (pose,vel,goal pose, paired agent goal pose) --> observations 
		self.obs_act_output_dim = 16
		self.final_input_dim = self.obs_act_output_dim #+ self.obs_input_dim #self.obs_z_output_dim + self.weight_input_dim
		self.final_output_dim = 1
		self.critic_network_V = ScalarDotProductCriticNetwork_V(self.obs_input_dim, self.obs_output_dim, self.obs_act_input_dim, self.obs_act_output_dim, self.final_input_dim, self.final_output_dim, self.num_agents, self.num_actions, self.softmax_cut_threshold).to(self.device)
		
		
		# SCALAR DOT PRODUCT POLICY NETWORK
		self.obs_input_dim = 2*3
		self.obs_output_dim = 16
		self.final_input_dim = self.obs_output_dim
		self.final_output_dim = self.num_actions
		self.policy_network = ScalarDotProductPolicyNetwork(self.obs_input_dim, self.obs_output_dim, self.final_input_dim, self.final_output_dim, self.num_agents, self.num_actions, self.softmax_cut_threshold).to(self.device)


		# MLP POLICY
		# self.policy_input_dim = 2*(3+2*(self.num_agents-1)) #2 for pose, 2 for vel and 2 for goal of current agent and rest (2 each) for relative position and relative velocity of other agents
		# self.policy_output_dim = self.env.action_space[0].n
		# policy_network_size = (self.policy_input_dim,512,256,self.policy_output_dim)
		# self.policy_network = PolicyNetwork(policy_network_size).to(self.device)


		# Loading models
		# model_path_value = "../../../models/Scalar_dot_product/collision_avoidance/4_Agents/SingleAttentionMechanism/with_prd_soft_adv/critic_networks/14-05-2021VN_ATN_FCN_lr0.01_PN_FCN_lr0.0002_GradNorm0.5_Entropy0.008_trace_decay0.98lambda_0.001topK_2select_above_threshold0.1softmax_cut_threshold0.1_epsiode29000.pt"
		# model_path_policy = "../../../models/Scalar_dot_product/collision_avoidance/4_Agents/SingleAttentionMechanism/with_prd_soft_adv/actor_networks/14-05-2021_PN_FCN_lr0.0002VN_SAT_FCN_lr0.01_GradNorm0.5_Entropy0.008_trace_decay0.98lambda_0.001topK_2select_above_threshold0.1softmax_cut_threshold0.1_epsiode29000.pt"
		# For CPU
		# self.critic_network.load_state_dict(torch.load(model_path_value,map_location=torch.device('cpu')))
		# self.policy_network.load_state_dict(torch.load(model_path_policy,map_location=torch.device('cpu')))
		# # For GPU
		# self.critic_network.load_state_dict(torch.load(model_path_value))
		# self.policy_network.load_state_dict(torch.load(model_path_policy))


		# self.critic_optimizer_Q = optim.Adam(self.critic_network_Q.parameters(),lr=self.value_lr)
		self.critic_optimizer_V = optim.Adam(self.critic_network_V.parameters(),lr=self.value_lr)
		self.policy_optimizer = optim.Adam(self.policy_network.parameters(),lr=self.policy_lr)


	def get_action(self,state):
		# MLP
		# state = torch.FloatTensor(state).to(self.device)
		# dists = self.policy_network.forward(state)
		# probs = Categorical(dists)
		# index = probs.sample().cpu().detach().item()

		# return index

		# GNN
		state = torch.FloatTensor([state]).to(self.device)
		dists, _ = self.policy_network.forward(state)
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
		
		





	def update(self,states_critic,next_states_critic,one_hot_actions,one_hot_next_actions,actions,states_actor,next_states_actor,rewards,dones):

		'''
		Getting the probability mass function over the action space for each agent
		'''
		# MLP
		# probs = self.policy_network.forward(actor_graphs).reshape(-1,self.num_agents,self.num_actions)
		# probs = self.policy_network.forward(states_actor)
		# next_probs = self.policy_network.forward(next_states_actor)

		# GNN
		probs, weight_policy = self.policy_network.forward(states_actor)

		'''
		Calculate V values
		'''
		# Q_values, weights_Q = self.critic_network_Q.forward(states_critic, probs.detach(), one_hot_actions)
		# V_values_next, _ = self.critic_network.forward(next_states_critic, next_probs.detach(), one_hot_next_actions)
		# Q_values_act_chosen = torch.sum(Q_values.reshape(-1,self.num_agents, self.num_agents, self.num_actions) * one_hot_actions.unsqueeze(-2), dim=-1)
		# V_values_next = V_values_next.reshape(-1,self.num_agents,self.num_agents)
		V_values_baseline, weights_V = self.critic_network_V.forward(states_critic, probs.detach(), one_hot_actions)
		V_values_baseline = V_values_baseline.reshape(-1,self.num_agents,self.num_agents)

		
	# # ***********************************************************************************
	# 	#update critic (value_net)
	# we need a TxNxN vector so inflate the discounted rewards by N --> cloning the discounted rewards for an agent N times
		discounted_rewards = self.calculate_returns(rewards,self.gamma).unsqueeze(-2).repeat(1,self.num_agents,1).to(self.device)
		discounted_rewards = torch.transpose(discounted_rewards,-1,-2)


		# BOOTSTRAP LOSS
		# target_values = torch.transpose(rewards.unsqueeze(-2).repeat(1,self.num_agents,1),-1,-2) + self.gamma*V_values_next*(1-dones.unsqueeze(-1))
		# value_loss = F.smooth_l1_loss(V_values,target_values)

		# MONTE CARLO LOSS
		# value_loss_Q = F.smooth_l1_loss(Q_values_act_chosen,discounted_rewards)
		value_loss_V = F.smooth_l1_loss(V_values_baseline,discounted_rewards)
		
		# # ***********************************************************************************
	# 	#update actor (policy net)
	# # ***********************************************************************************
		entropy = -torch.mean(torch.sum(probs * torch.log(torch.clamp(probs, 1e-10,1.0)), dim=2))
		

		# advantage = torch.sum(Q_values_act_chosen - V_values_baseline, dim=-2)
		# advantage = torch.sum(self.calculate_advantages(discounted_rewards, V_values_baseline, rewards, dones, True, False), dim=-2)

		values, indices = torch.topk(weights_V,k=1,dim=-1)
		masking_advantage = torch.transpose(torch.sum(F.one_hot(indices, num_classes=self.num_agents), dim=-2),-1,-2)
		advantage = torch.sum(self.calculate_advantages(discounted_rewards, V_values_baseline, rewards, dones, True, False) * masking_advantage,dim=-2)


		probs = Categorical(probs)
		policy_loss = -probs.log_prob(actions) * advantage.detach()
		policy_loss = policy_loss.mean() - self.entropy_pen*entropy
		# policy_loss = torch.Tensor([0])
	# # ***********************************************************************************
		
	# **********************************
		# self.critic_optimizer_Q.zero_grad()
		# value_loss_Q.backward(retain_graph=False)
		# grad_norm_value_Q = torch.nn.utils.clip_grad_norm_(self.critic_network_Q.parameters(),0.5)
		# self.critic_optimizer_Q.step()

		self.critic_optimizer_V.zero_grad()
		value_loss_V.backward(retain_graph=False)
		grad_norm_value_V = torch.nn.utils.clip_grad_norm_(self.critic_network_V.parameters(),0.5)
		self.critic_optimizer_V.step()


		self.policy_optimizer.zero_grad()
		policy_loss.backward(retain_graph=False)
		grad_norm_policy = torch.nn.utils.clip_grad_norm_(self.policy_network.parameters(),0.5)
		self.policy_optimizer.step()
		# grad_norm_policy = 0

		# return value_loss_Q, value_loss_V, policy_loss, entropy, grad_norm_value_Q, grad_norm_value_V, grad_norm_policy, weights_Q, weights_V, weight_policy
		return value_loss_V, policy_loss, entropy, grad_norm_value_V, grad_norm_policy, weights_V, weight_policy