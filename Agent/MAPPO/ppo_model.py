from typing import Any, List, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import datetime
import math

class RolloutBuffer:
	def __init__(self):
		self.actions = []
		self.one_hot_actions = []
		self.probs = []
		self.states_critic = []
		self.states_actor = []
		self.logprobs = []
		self.rewards = []
		self.dones = []
	

	def clear(self):
		del self.actions[:]
		del self.states_critic[:]
		del self.states_actor[:]
		del self.probs[:]
		del self.one_hot_actions[:]
		del self.logprobs[:]
		del self.rewards[:]
		del self.dones[:]


class MLPPolicy(nn.Module):
	def __init__(self,current_agent_state_dim, other_agent_state_dim, num_agents, action_dim, device):
		super(MLPPolicy,self).__init__()

		self.name = "MLPPolicy"
		self.num_agents = num_agents		
		self.device = device

		self.Policy = nn.Sequential(
			nn.Linear(current_agent_state_dim+(other_agent_state_dim*num_agents),128),
			nn.LeakyReLU(),
			nn.Linear(128,64),
			nn.LeakyReLU(),
			nn.Linear(64,action_dim),
			nn.Softmax(dim=-1)
			)

	def forward(self, states):

		# T x num_agents x state_dim
		T = states.shape[0]
		
		# # [s0;s1;s2;s3]  -> [s0 s1 s2 s3; s1 s2 s3 s0; s2 s3 s1 s0 ....]

		# states_aug = [torch.roll(states,i,1) for i in range(self.num_agents)]

		# states_aug = torch.cat(states_aug,dim=2)

		Policy = self.Policy(states)

		return Policy, 1/self.num_agents*torch.ones((T,self.num_agents,self.num_agents),device=self.device)


class TransformerPolicy(nn.Module):
	def __init__(self, obs_input_dim, final_output_dim, num_agents, num_actions, num_heads, device):
		super(TransformerPolicy, self).__init__()
		
		self.name = "TransformerPolicy"
		self.num_heads = num_heads

		self.num_agents = num_agents
		self.num_actions = num_actions
		self.device = device

		obs_output_dim = 128//self.num_heads

		self.state_embed_list = []
		self.key_list = []
		self.query_list = []
		self.attention_value_list = []
		for i in range(self.num_heads):
			self.state_embed_list.append(nn.Sequential(nn.Linear(obs_input_dim, 128), nn.LeakyReLU()).to(self.device))
			self.key_list.append(nn.Linear(128, obs_output_dim, bias=True).to(self.device))
			self.query_list.append(nn.Linear(128, obs_output_dim, bias=True).to(self.device))
			self.attention_value_list.append(nn.Sequential(nn.Linear(128, obs_output_dim, bias=True), nn.LeakyReLU()).to(self.device))

		self.d_k = obs_output_dim
		# ********************************************************************************************************

		# ********************************************************************************************************
		# FCN FINAL LAYER TO GET VALUES
		final_input_dim = obs_output_dim*self.num_heads
		self.final_policy_layers = nn.Sequential(
			nn.Linear(final_input_dim, 64, bias=True), 
			nn.LeakyReLU(),
			nn.Linear(64, final_output_dim, bias=True)
			)
		# ********************************************************************************************************


		self.reset_parameters()


	def reset_parameters(self):
		"""Reinitialize learnable parameters."""
		gain_leaky = nn.init.calculate_gain('leaky_relu')

		for i in range(self.num_heads):
			nn.init.xavier_uniform_(self.state_embed_list[i][0].weight, gain=gain_leaky)

			nn.init.xavier_uniform_(self.key_list[i].weight)
			nn.init.xavier_uniform_(self.query_list[i].weight)
			nn.init.xavier_uniform_(self.attention_value_list[i][0].weight)


		nn.init.xavier_uniform_(self.final_policy_layers[0].weight, gain=gain_leaky)
		nn.init.xavier_uniform_(self.final_policy_layers[2].weight, gain=gain_leaky)



	def forward(self, states):
		weights = []
		node_features = []
		for i in range(self.num_heads):
			# EMBED STATES
			states_embed = self.state_embed_list[i](states)
			# KEYS
			key_obs = self.key_list[i](states_embed)
			# QUERIES
			query_obs = self.query_list[i](states_embed)
			# WEIGHT
			weight = F.softmax(torch.matmul(query_obs,key_obs.transpose(1,2))/math.sqrt(self.d_k),dim=-1)
			weights.append(weight)

			attention_values = self.attention_value_list[i](states_embed)
			node_feature = torch.matmul(weight, attention_values)

			node_features.append(node_feature)

		node_features = torch.cat(node_features, dim=-1).to(self.device)
		Policy = F.softmax(self.final_policy_layers(node_features), dim=-1)

		return Policy, weights



class DualTransformerPolicy(nn.Module):
	def __init__(self, obs_input_dim, final_output_dim, num_agents, num_actions, num_heads_1, num_heads_2, device):
		super(DualTransformerPolicy, self).__init__()
		
		self.name = "DualTransformerPolicy"
		self.num_heads_1 = num_heads_1
		self.num_heads_2 = num_heads_2

		self.num_agents = num_agents
		self.num_actions = num_actions
		self.device = device

		output_dim = 128//self.num_heads_1

		self.state_embed_list_1 = []
		self.key_list_1 = []
		self.query_list_1 = []
		self.attention_value_list_1 = []
		for i in range(self.num_heads_1):
			self.state_embed_list_1.append(nn.Sequential(nn.Linear(obs_input_dim, 128), nn.LeakyReLU()).to(self.device))
			self.key_list_1.append(nn.Linear(128, 128, bias=True).to(self.device))
			self.query_list_1.append(nn.Linear(128, 128, bias=True).to(self.device))
			self.attention_value_list_1.append(nn.Sequential(nn.Linear(128, output_dim, bias=True), nn.LeakyReLU()).to(self.device))

		self.d_k_1 = 128

		input_dim = output_dim*self.num_heads_1
		obs_output_dim = 128//self.num_heads_2

		self.state_embed_list_2 = []
		self.key_list_2 = []
		self.query_list_2 = []
		self.attention_value_list_2 = []
		for i in range(self.num_heads_2):
			self.state_embed_list_2.append(nn.Sequential(nn.Linear(input_dim, 128), nn.LeakyReLU()).to(self.device))
			self.key_list_2.append(nn.Linear(128, 128, bias=True).to(self.device))
			self.query_list_2.append(nn.Linear(128, 128, bias=True).to(self.device))
			self.attention_value_list_2.append(nn.Sequential(nn.Linear(128, obs_output_dim, bias=True), nn.LeakyReLU()).to(self.device))

		self.d_k_2 = 128
		# ********************************************************************************************************

		# ********************************************************************************************************
		# FCN FINAL LAYER TO GET VALUES
		final_input_dim = obs_output_dim*self.num_heads_2
		self.final_policy_layers = nn.Sequential(
			nn.Linear(final_input_dim, 64, bias=True), 
			nn.LeakyReLU(),
			nn.Linear(64, final_output_dim, bias=True)
			)
		# ********************************************************************************************************


		self.reset_parameters()


	def reset_parameters(self):
		"""Reinitialize learnable parameters."""
		gain_leaky = nn.init.calculate_gain('leaky_relu')

		for i in range(self.num_heads_1):
			nn.init.xavier_uniform_(self.state_embed_list_1[i][0].weight, gain=gain_leaky)

			nn.init.xavier_uniform_(self.key_list_1[i].weight)
			nn.init.xavier_uniform_(self.query_list_1[i].weight)
			nn.init.xavier_uniform_(self.attention_value_list_1[i][0].weight)

		for i in range(self.num_heads_2):
			nn.init.xavier_uniform_(self.state_embed_list_2[i][0].weight, gain=gain_leaky)

			nn.init.xavier_uniform_(self.key_list_2[i].weight)
			nn.init.xavier_uniform_(self.query_list_2[i].weight)
			nn.init.xavier_uniform_(self.attention_value_list_2[i][0].weight)


		nn.init.xavier_uniform_(self.final_policy_layers[0].weight, gain=gain_leaky)
		nn.init.xavier_uniform_(self.final_policy_layers[2].weight, gain=gain_leaky)



	def forward(self, states):
		weights_1 = []
		attention_values_list = []
		for i in range(self.num_heads_1):
			# EMBED STATES
			states_embed = self.state_embed_list_1[i](states)
			# KEYS
			key_obs = self.key_list_1[i](states_embed)
			# QUERIES
			query_obs = self.query_list_1[i](states_embed)
			# WEIGHT
			weight = F.softmax(torch.matmul(query_obs,key_obs.transpose(1,2))/math.sqrt(self.d_k_1),dim=-1)
			weights_1.append(weight)

			attention_values = self.attention_value_list_1[i](states_embed)
			attention_values = torch.matmul(weight, attention_values)

			attention_values_list.append(attention_values)

		attention_values_list = torch.cat(attention_values, dim=-1).to(self.device)

		weights_2 = []
		node_features = []
		for i in range(self.num_heads_2):
			# EMBED STATES
			states_embed = self.state_embed_list_2[i](attention_values_list)
			# KEYS
			key_obs = self.key_list_2[i](states_embed)
			# QUERIES
			query_obs = self.query_list_2[i](states_embed)
			# WEIGHT
			weight = F.softmax(torch.matmul(query_obs,key_obs.transpose(1,2))/math.sqrt(self.d_k_2),dim=-1)
			weights_2.append(weight)

			attention_values = self.attention_value_list_2[i](states_embed)
			node_feature = torch.matmul(weight, attention_values)

			node_features.append(node_feature)


		node_features = torch.cat(node_features, dim=-1).to(self.device)
		Policy = F.softmax(self.final_policy_layers(node_features), dim=-1)

		return Policy, weights_1, weights_2



class TransformerCritic(nn.Module):
	'''
	https://proceedings.neurips.cc/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf
	'''
	def __init__(self, obs_input_dim, final_output_dim, num_agents, num_actions, num_heads, device):
		super(TransformerCritic, self).__init__()
		
		self.name = "TransformerCritic"

		self.num_agents = num_agents
		self.num_actions = num_actions
		self.device = device
		self.num_heads = num_heads

		self.state_embed_list = []
		self.key_list = []
		self.query_list = []
		self.state_act_pol_embed_list = []
		self.attention_value_list = []

		obs_output_dim = 128
		obs_act_input_dim = obs_input_dim+self.num_actions
		obs_act_output_dim = 128//self.num_heads

		for i in range(self.num_heads):
			self.state_embed_list.append(nn.Sequential(nn.Linear(obs_input_dim, 128), nn.LeakyReLU()).to(self.device))
			self.key_list.append(nn.Linear(128, obs_output_dim, bias=True).to(self.device))
			self.query_list.append(nn.Linear(128, obs_output_dim, bias=True).to(self.device))
			self.state_act_pol_embed_list.append(nn.Sequential(nn.Linear(obs_act_input_dim, 128, bias=True), nn.LeakyReLU()).to(self.device))
			self.attention_value_list.append(nn.Sequential(nn.Linear(128, obs_act_output_dim), nn.LeakyReLU()).to(self.device))

		# dimesion of key
		self.d_k = obs_output_dim

		# ********************************************************************************************************

		# ********************************************************************************************************
		final_input_dim = obs_act_output_dim*self.num_heads
		# FCN FINAL LAYER TO GET VALUES
		self.final_value_layers = nn.Sequential(
			nn.Linear(final_input_dim, 64, bias=True), 
			nn.LeakyReLU(),
			nn.Linear(64, final_output_dim, bias=True)
			)
		# ********************************************************************************************************	

		self.place_policies = torch.zeros(self.num_agents,self.num_agents,obs_act_input_dim).to(self.device)
		self.place_actions = torch.ones(self.num_agents,self.num_agents,obs_act_input_dim).to(self.device)
		one_hots = torch.ones(obs_act_input_dim)
		zero_hots = torch.zeros(obs_act_input_dim)

		for j in range(self.num_agents):
			self.place_policies[j][j] = one_hots
			self.place_actions[j][j] = zero_hots


		self.reset_parameters()


	def reset_parameters(self):
		"""Reinitialize learnable parameters."""
		gain_leaky = nn.init.calculate_gain('leaky_relu')

		for i in range(self.num_heads):
			nn.init.xavier_uniform_(self.state_embed_list[i][0].weight, gain=gain_leaky)
			nn.init.xavier_uniform_(self.state_act_pol_embed_list[i][0].weight, gain=gain_leaky)

			nn.init.xavier_uniform_(self.key_list[i].weight)
			nn.init.xavier_uniform_(self.query_list[i].weight)
			nn.init.xavier_uniform_(self.attention_value_list[i][0].weight)


		nn.init.xavier_uniform_(self.final_value_layers[0].weight, gain=gain_leaky)
		nn.init.xavier_uniform_(self.final_value_layers[2].weight, gain=gain_leaky)



	def forward(self, states, policies, actions):

		obs_actions = torch.cat([states,actions],dim=-1)
		obs_policy = torch.cat([states,policies], dim=-1)
		obs_actions = obs_actions.repeat(1,self.num_agents,1).reshape(obs_actions.shape[0],self.num_agents,self.num_agents,-1)
		obs_policy = obs_policy.repeat(1,self.num_agents,1).reshape(obs_policy.shape[0],self.num_agents,self.num_agents,-1)
		obs_actions_policies = self.place_policies*obs_policy + self.place_actions*obs_actions

		node_features = []
		weights = []
		for i in range(self.num_heads):
			# EMBED STATES
			states_embed = self.state_embed_list[i](states)
			# KEYS
			key_obs = self.key_list[i](states_embed)
			# QUERIES
			query_obs = self.query_list[i](states_embed)
			# WEIGHT
			weight = F.softmax(torch.matmul(query_obs,key_obs.transpose(1,2))/math.sqrt(self.d_k),dim=-1)
			weights.append(weight)

			# EMBED STATE ACTION POLICY
			obs_actions_policies_embed = self.state_act_pol_embed_list[i](obs_actions_policies)
			attention_values = self.attention_value_list[i](obs_actions_policies_embed)
			attention_values = attention_values.repeat(1,self.num_agents,1,1).reshape(attention_values.shape[0],self.num_agents,self.num_agents,self.num_agents,-1)
			
			weight = weight.unsqueeze(-2).repeat(1,1,self.num_agents,1).unsqueeze(-1)
			weighted_attention_values = attention_values*weight
			node_feature = torch.sum(weighted_attention_values, dim=-2)
			node_features.append(node_feature)

		node_features = torch.cat(node_features, dim=-1).to(self.device)
		
		Value = self.final_value_layers(node_features)

		return Value, weights


class DualTransformerCritic(nn.Module):
	def __init__(self, obs_input_dim, final_output_dim, num_agents, num_actions, num_heads_1, num_heads_2, device):
		super(DualTransformerCritic, self).__init__()
		
		self.name = "DualTransformerCritic"

		self.num_agents = num_agents
		self.num_actions = num_actions
		self.device = device
		self.num_heads_1 = num_heads_1
		self.num_heads_2 = num_heads_2

		obs_output_dim = 128//self.num_heads_1

		# TRANSFORMER LAYER 1
		self.state_embed_list_1 = []
		self.key_list_1 = []
		self.query_list_1 = []
		self.attention_value_list_1 = []
		for i in range(self.num_heads_1):
			self.state_embed_list_1.append(nn.Sequential(nn.Linear(obs_input_dim, 128), nn.LeakyReLU()).to(self.device))
			self.key_list_1.append(nn.Linear(128, 128, bias=True).to(self.device))
			self.query_list_1.append(nn.Linear(128, 128, bias=True).to(self.device))
			self.attention_value_list_1.append(nn.Sequential(nn.Linear(128, obs_output_dim, bias=True), nn.LeakyReLU()).to(self.device))

		self.d_k_1 = 128

		obs_output_dim_ = 128
		obs_act_input_dim = obs_input_dim+self.num_actions
		obs_act_output_dim = 128//self.num_heads_2
		# TRANSFORMER LAYER 2
		self.state_embed_list_2 = []
		self.key_list_2 = []
		self.query_list_2 = []
		self.attention_value_list_2 = []
		for i in range(self.num_heads_2):
			self.state_embed_list_2.append(nn.Sequential(nn.Linear(obs_output_dim*self.num_heads_1, 128), nn.LeakyReLU()).to(self.device))
			self.key_list_2.append(nn.Linear(128, obs_output_dim_, bias=True).to(self.device))
			self.query_list_2.append(nn.Linear(128, obs_output_dim_, bias=True).to(self.device))
			self.state_act_pol_embed_list_2.append(nn.Sequential(nn.Linear(obs_act_input_dim, 128, bias=True), nn.LeakyReLU()).to(self.device))
			self.attention_value_list_2.append(nn.Sequential(nn.Linear(128, obs_act_output_dim, bias=True), nn.LeakyReLU()).to(self.device))
		# dimesion of key
		self.d_k_2 = obs_output_dim_  

		# ********************************************************************************************************

		# ********************************************************************************************************
		final_input_dim = obs_act_output_dim*self.num_heads_2
		# FCN FINAL LAYER TO GET VALUES
		self.final_value_layers = nn.Sequential(
			nn.Linear(final_input_dim, 64, bias=True), 
			nn.LeakyReLU(),
			nn.Linear(64, final_output_dim, bias=True)
			)
		# ********************************************************************************************************	

		self.place_policies = torch.zeros(self.num_agents,self.num_agents,obs_act_input_dim).to(self.device)
		self.place_actions = torch.ones(self.num_agents,self.num_agents,obs_act_input_dim).to(self.device)
		one_hots = torch.ones(obs_act_input_dim)
		zero_hots = torch.zeros(obs_act_input_dim)

		for j in range(self.num_agents):
			self.place_policies[j][j] = one_hots
			self.place_actions[j][j] = zero_hots


		self.reset_parameters()


	def reset_parameters(self):
		"""Reinitialize learnable parameters."""
		gain_leaky = nn.init.calculate_gain('leaky_relu')

		for i in range(self.num_heads_1):
			nn.init.xavier_uniform_(self.state_embed_list_1[i][0].weight, gain=gain_leaky)

			nn.init.xavier_uniform_(self.key_list_1[i].weight)
			nn.init.xavier_uniform_(self.query_list[i].weight)
			nn.init.xavier_uniform_(self.attention_value_list[i][0].weight)

		for i in range(self.num_heads_2):
			nn.init.xavier_uniform_(self.state_embed_list_2[i][0].weight, gain=gain_leaky)
			nn.init.xavier_uniform_(self.state_act_pol_embed_list_2[i][0].weight, gain=gain_leaky)

			nn.init.xavier_uniform_(self.key_list_2[i].weight)
			nn.init.xavier_uniform_(self.query_list_2[i].weight)
			nn.init.xavier_uniform_(self.attention_value_list_2[i][0].weight)


		nn.init.xavier_uniform_(self.final_value_layers[0].weight, gain=gain_leaky)
		nn.init.xavier_uniform_(self.final_value_layers[2].weight, gain=gain_leaky)



	def forward(self, states, policies, actions):
		obs_actions = torch.cat([states,actions],dim=-1)
		obs_policy = torch.cat([states,policies], dim=-1)
		obs_actions = obs_actions.repeat(1,self.num_agents,1).reshape(obs_actions.shape[0],self.num_agents,self.num_agents,-1)
		obs_policy = obs_policy.repeat(1,self.num_agents,1).reshape(obs_policy.shape[0],self.num_agents,self.num_agents,-1)
		obs_actions_policies = self.place_policies*obs_policy + self.place_actions*obs_actions

		attention_values_1 = []
		weights_1 = []
		for i in range(self.num_heads_1):
			# EMBED STATES PREPROC
			states_embed_preproc = self.state_embed_list_1[i](states)
			# KEYS
			key_obs_preproc = self.key_list_1[i](states_embed_preproc)
			# QUERIES
			query_obs_preproc = self.query_list_1[i](states_embed_preproc)
			# WEIGHT
			weight_preproc = F.softmax(torch.matmul(query_obs_preproc,key_obs_preproc.transpose(1,2))/math.sqrt(self.d_k_1),dim=-1)
			weights_1.append(weight_preproc)
			# ATTENTION VALUES
			attention_values_preproc = self.attention_value_list_1[i](states_embed_preproc)
			attention_values_preproc = torch.matmul(weight_preproc, states_embed_preproc)
			attention_values_1.append(attention_values_preproc)

		attention_values_1 = torch.cat(attention_values_1, dim=-1).to(self.device)

		weights_2 = []
		node_features = []
		for i in range(self.num_heads_2):
			# EMBED STATES
			states_embed = self.state_embed_list_2[i](attention_values_1)
			# KEYS
			key_obs = self.key_list_2[i](states_embed)
			# QUERIES
			query_obs = self.query_list_2[i](states_embed)
			# WEIGHT
			weight = F.softmax(torch.matmul(query_obs,key_obs.transpose(1,2))/math.sqrt(self.d_k_2),dim=-1)
			weights_2.append(weight)

		
			# EMBED STATE ACTION POLICY
			obs_actions_policies_embed = self.state_act_pol_embed_list_2[i](obs_actions_policies)
			attention_values = self.attention_value_list_2[i](obs_actions_policies_embed)
			attention_values = attention_values.repeat(1,self.num_agents,1,1).reshape(attention_values.shape[0],self.num_agents,self.num_agents,self.num_agents,-1)
			
			weight = weight.unsqueeze(-2).repeat(1,1,self.num_agents,1).unsqueeze(-1)
			weighted_attention_values = attention_values*weight
			node_feature = torch.sum(weighted_attention_values, dim=-2)

			node_features.append(node_feature)

		node_features = torch.cat(node_features, dim=-1).to(self.device)
		Value = self.final_value_layers(node_features)

		return Value, weight_preproc, ret_weight



class TransformerCritic_threshold_pred(nn.Module):
	'''
	https://proceedings.neurips.cc/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf
	'''
	def __init__(self, obs_input_dim, final_output_dim, num_agents, num_actions, num_heads, device):
		super(TransformerCritic_threshold_pred, self).__init__()
		
		self.name = "TransformerCritic_threshold_pred"

		self.num_agents = num_agents
		self.num_actions = num_actions
		self.device = device
		self.num_heads = num_heads

		self.state_embed_threshold = []
		self.state_embed_list = []
		self.key_threshold = []
		self.key_list = []
		self.query_threshold = []
		self.query_list = []
		self.state_act_pol_embed_list = []
		self.attention_value_list = []

		obs_output_dim = 128
		obs_act_input_dim = obs_input_dim+self.num_actions
		obs_act_output_dim = 128//self.num_heads
		for i in range(self.num_heads):
			self.state_embed_threshold.append(nn.Sequential(nn.Linear(obs_input_dim, 128), nn.LeakyReLU()).to(self.device))
			self.state_embed_list.append(nn.Sequential(nn.Linear(obs_input_dim, 128), nn.LeakyReLU()).to(self.device))
			self.key_list.append(nn.Linear(128, obs_output_dim, bias=True).to(self.device))
			self.query_list.append(nn.Linear(128, obs_output_dim, bias=True).to(self.device))
			self.key_threshold.append(nn.Linear(128, obs_output_dim, bias=True).to(self.device))
			self.query_threshold.append(nn.Linear(128, obs_output_dim, bias=True).to(self.device))
			self.state_act_pol_embed_list.append(nn.Sequential(nn.Linear(obs_act_input_dim, 128, bias=True), nn.LeakyReLU()).to(self.device))
			self.attention_value_list.append(nn.Sequential(nn.Linear(128, obs_act_output_dim), nn.LeakyReLU()).to(self.device))

		# dimesion of key
		self.d_k = obs_output_dim
		self.d_k_threshold = obs_output_dim

		# ********************************************************************************************************

		# ********************************************************************************************************
		# FCN FINAL LAYER TO GET VALUES
		final_input_dim = obs_act_output_dim*self.num_heads
		self.final_value_layers = nn.Sequential(
			nn.Linear(final_input_dim, 64, bias=True), 
			nn.LeakyReLU(),
			nn.Linear(64, final_output_dim, bias=True)
			)
		# ********************************************************************************************************	

		self.place_policies = torch.zeros(self.num_agents,self.num_agents,obs_act_input_dim).to(self.device)
		self.place_actions = torch.ones(self.num_agents,self.num_agents,obs_act_input_dim).to(self.device)
		one_hots = torch.ones(obs_act_input_dim)
		zero_hots = torch.zeros(obs_act_input_dim)

		for j in range(self.num_agents):
			self.place_policies[j][j] = one_hots
			self.place_actions[j][j] = zero_hots


		self.reset_parameters()


	def reset_parameters(self):
		"""Reinitialize learnable parameters."""
		gain_leaky = nn.init.calculate_gain('leaky_relu')

		for i in range(self.num_heads):
			nn.init.xavier_uniform_(self.state_embed_list[i][0].weight, gain=gain_leaky)
			nn.init.xavier_uniform_(self.state_act_pol_embed_list[i][0].weight, gain=gain_leaky)

			nn.init.xavier_uniform_(self.key_list[i].weight)
			nn.init.xavier_uniform_(self.query_list[i].weight)
			nn.init.xavier_uniform_(self.attention_value_list[i][0].weight)

			nn.init.xavier_uniform_(self.state_embed_threshold[i][0].weight, gain=gain_leaky)
			nn.init.xavier_uniform_(self.key_threshold[i].weight)
			nn.init.xavier_uniform_(self.query_threshold[i].weight)


		nn.init.xavier_uniform_(self.final_value_layers[0].weight, gain=gain_leaky)
		nn.init.xavier_uniform_(self.final_value_layers[2].weight, gain=gain_leaky)



	def forward(self, states, policies, actions):

		obs_actions = torch.cat([states,actions],dim=-1)
		obs_policy = torch.cat([states,policies], dim=-1)
		obs_actions = obs_actions.repeat(1,self.num_agents,1).reshape(obs_actions.shape[0],self.num_agents,self.num_agents,-1)
		obs_policy = obs_policy.repeat(1,self.num_agents,1).reshape(obs_policy.shape[0],self.num_agents,self.num_agents,-1)
		obs_actions_policies = self.place_policies*obs_policy + self.place_actions*obs_actions

		node_features = []
		weights = []
		for i in range(self.num_heads):
			# EMBED STATES
			states_embed = self.state_embed_list[i](states)
			# KEYS
			key_obs = self.key_list[i](states_embed)
			# QUERIES
			query_obs = self.query_list[i](states_embed)
			# WEIGHT
			weight = F.softmax(torch.matmul(query_obs,key_obs.transpose(1,2))/math.sqrt(self.d_k),dim=-1)

			state_embed_threshold = self.state_embed_threshold[i](states)
			query_threshold = self.query_threshold[i](state_embed_threshold)
			key_threshold = self.key_threshold[i](torch.sum(state_embed_threshold, dim=-2)).unsqueeze(-2).repeat(1,self.num_agents,1)
			score_threshold = torch.matmul(query_threshold,key_threshold.transpose(1,2))/math.sqrt(self.d_k_threshold)
			# threshold = (torch.tanh(score_threshold)+1)/2.0
			threshold = torch.sigmoid(score_threshold)
			print("Threshold", torch.mean(threshold).item())
			weight_diff = torch.relu(weight-threshold*0.05)
			weight = torch.div(weight_diff,torch.sum(weight_diff+1e-12,dim=-1).unsqueeze(-1).repeat(1,1,self.num_agents))
			weights.append(weight)

			# EMBED STATE ACTION POLICY
			obs_actions_policies_embed = self.state_act_pol_embed_list[i](obs_actions_policies)
			attention_values = self.attention_value_list[i](obs_actions_policies_embed)
			attention_values = attention_values.repeat(1,self.num_agents,1,1).reshape(attention_values.shape[0],self.num_agents,self.num_agents,self.num_agents,-1)
			
			weight = weight.unsqueeze(-2).repeat(1,1,self.num_agents,1).unsqueeze(-1)
			weighted_attention_values = attention_values*weight
			node_feature = torch.sum(weighted_attention_values, dim=-2)
			node_features.append(node_feature)

		node_features = torch.cat(node_features, dim=-1).to(self.device)
		Value = self.final_value_layers(node_features)

		return Value, weights, threshold