from typing import Any, List, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import datetime
import math


class ScalarDotProductCriticNetwork(nn.Module):
	def __init__(self, obs_input_dim, obs_output_dim, obs_act_input_dim, obs_act_output_dim, final_input_dim, final_output_dim, num_agents, num_actions, threshold=0.1):
		super(ScalarDotProductCriticNetwork, self).__init__()
		
		self.num_agents = num_agents
		self.num_actions = num_actions
		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		# self.device = "cpu"

		self.key_layer = nn.Linear(obs_input_dim, obs_output_dim, bias=False)

		self.query_layer = nn.Linear(obs_input_dim, obs_output_dim, bias=False)

		self.attention_value_layer = nn.Linear(obs_act_input_dim, obs_act_output_dim, bias=False)

		# dimesion of key
		self.d_k_obs_act = obs_output_dim

		# NOISE
		self.noise_normal = torch.distributions.Normal(loc=torch.tensor([0.0]), scale=torch.tensor([1.0]))
		self.noise_uniform = torch.rand
		# ********************************************************************************************************

		# ********************************************************************************************************
		# FCN FINAL LAYER TO GET VALUES
		self.final_value_layer_1 = nn.Linear(final_input_dim, 64, bias=False)
		self.final_value_layer_2 = nn.Linear(64, final_output_dim, bias=False)
		# ********************************************************************************************************	

		self.place_policies = torch.zeros(self.num_agents,self.num_agents,obs_act_input_dim).to(self.device)
		self.place_actions = torch.ones(self.num_agents,self.num_agents,obs_act_input_dim).to(self.device)
		one_hots = torch.ones(obs_act_input_dim)
		zero_hots = torch.zeros(obs_act_input_dim)

		for j in range(self.num_agents):
			self.place_policies[j][j] = one_hots
			self.place_actions[j][j] = zero_hots

		self.threshold = threshold
		self.obs_act_input_dim = obs_act_input_dim
		# ********************************************************************************************************* 

		self.reset_parameters()


	def mixing_actions_policies(self):
		self.place_policies = torch.zeros(self.num_agents,self.num_agents,self.obs_act_input_dim).to(self.device)
		self.place_actions = torch.ones(self.num_agents,self.num_agents,self.obs_act_input_dim).to(self.device)
		one_hots = torch.ones(self.obs_act_input_dim)
		zero_hots = torch.zeros(self.obs_act_input_dim)

		for j in range(self.num_agents):
			self.place_policies[j][j] = one_hots
			self.place_actions[j][j] = zero_hots


	def reset_parameters(self):
		"""Reinitialize learnable parameters."""
		gain_leaky = nn.init.calculate_gain('leaky_relu')

		nn.init.xavier_uniform_(self.key_layer.weight)
		nn.init.xavier_uniform_(self.query_layer.weight)
		nn.init.xavier_uniform_(self.attention_value_layer.weight)


		nn.init.xavier_uniform_(self.final_value_layer_1.weight, gain=gain_leaky)
		nn.init.xavier_uniform_(self.final_value_layer_2.weight, gain=gain_leaky)



	def forward(self, states, policies, actions):

		# input to KEY, QUERY and ATTENTION VALUE NETWORK
		obs_actions = torch.cat([states,actions],dim=-1)
		# print("OBSERVATIONS ACTIONS")
		# print(obs_actions)

		# For calculating the right advantages
		obs_policy = torch.cat([states,policies], dim=-1)

		# KEYS
		key_obs_actions = self.key_layer(states)

		# QUERIES
		query_obs_actions = self.query_layer(states)

		score_obs_actions = torch.bmm(query_obs_actions,key_obs_actions.transpose(1,2)).reshape(-1,1)

		score_obs_actions = score_obs_actions.reshape(-1,self.num_agents,1)


		weight = F.softmax(score_obs_actions/math.sqrt(self.d_k_obs_act), dim=-2)
		weight = weight.reshape(weight.shape[0]//self.num_agents,self.num_agents,-1)
		ret_weight = weight
		
		obs_actions = obs_actions.repeat(1,self.num_agents,1).reshape(obs_actions.shape[0],self.num_agents,self.num_agents,-1)

		obs_policy = obs_policy.repeat(1,self.num_agents,1).reshape(obs_policy.shape[0],self.num_agents,self.num_agents,-1)

		# RANDOMIZING NUMBER OF AGENTS
		# self.mixing_actions_policies()

		obs_actions_policies = self.place_policies*obs_policy + self.place_actions*obs_actions

		# embedding the observation_actions_policies

		attention_values = torch.tanh(self.attention_value_layer(obs_actions_policies))

		attention_values = attention_values.repeat(1,self.num_agents,1,1).reshape(attention_values.shape[0],self.num_agents,self.num_agents,self.num_agents,-1)

		# SOFTMAX
		weight = weight.unsqueeze(-2).repeat(1,1,self.num_agents,1).unsqueeze(-1)
		weighted_attention_values = attention_values*weight

		# SOFTMAX WITH NOISE
		# weight = weight.unsqueeze(-2).repeat(1,1,self.num_agents,1).unsqueeze(-1)
		# uniform_noise = (self.noise_uniform((attention_values.view(-1).size())).reshape(attention_values.size()) - 0.5) * 0.1 #SCALING NOISE AND MAKING IT ZERO CENTRIC
		# weighted_attention_values = attention_values*weight + uniform_noise

		# SOFTMAX WITH NORMALIZATION
		# scaling_weight = F.relu(weight - self.threshold)
		# scaling_weight = torch.div(scaling_weight,torch.sum(scaling_weight,dim =-1).unsqueeze(-1))
		# ret_weight = scaling_weight
		# scaling_weight = scaling_weight.unsqueeze(-2).repeat(1,1,self.num_agents,1).unsqueeze(-1)
		# weighted_attention_values = attention_values*scaling_weight

		node_features = torch.sum(weighted_attention_values, dim=-2)

		Value = F.leaky_relu(self.final_value_layer_1(node_features))
		Value = self.final_value_layer_2(Value)

		return Value, ret_weight
		


class ScalarDotProductPolicyNetwork(nn.Module):
	def __init__(self, obs_input_dim, obs_output_dim, final_input_dim, final_output_dim, num_agents, num_actions, threshold=0.1):
		super(ScalarDotProductPolicyNetwork, self).__init__()
		
		self.num_agents = num_agents
		self.num_actions = num_actions
		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		# self.device = "cpu"

		self.key_layer = nn.Linear(obs_input_dim, obs_output_dim, bias=False)

		self.query_layer = nn.Linear(obs_input_dim, obs_output_dim, bias=False)

		self.attention_value_layer = nn.Linear(obs_input_dim, obs_output_dim, bias=False)

		# dimesion of key
		self.d_k_obs = obs_output_dim

		# NOISE
		self.noise_normal = torch.distributions.Normal(loc=torch.tensor([0.0]), scale=torch.tensor([1.0]))
		self.noise_uniform = torch.rand
		# ********************************************************************************************************

		# ********************************************************************************************************
		# FCN FINAL LAYER TO GET VALUES
		self.final_policy_layer_1 = nn.Linear(final_input_dim, 64, bias=False)
		self.final_policy_layer_2 = nn.Linear(64, final_output_dim, bias=False)
		# ********************************************************************************************************	

		self.threshold = threshold
		# ********************************************************************************************************* 

		self.reset_parameters()

	def reset_parameters(self):
		"""Reinitialize learnable parameters."""
		gain = nn.init.calculate_gain('leaky_relu')

		nn.init.xavier_uniform_(self.key_layer.weight)
		nn.init.xavier_uniform_(self.query_layer.weight)
		nn.init.xavier_uniform_(self.attention_value_layer.weight)


		nn.init.xavier_uniform_(self.final_policy_layer_1.weight, gain=gain)
		nn.init.xavier_uniform_(self.final_policy_layer_2.weight, gain=gain)



	def forward(self, states):
		
		# KEYS
		key_obs = self.key_layer(states)
		# QUERIES
		query_obs = self.query_layer(states)
		# SCORE CALCULATION
		score_obs = torch.bmm(query_obs,key_obs.transpose(1,2)).reshape(-1,1)
		score_obs = score_obs.reshape(-1,self.num_agents,1)
		# WEIGHT
		weight = F.softmax(score_obs/math.sqrt(self.d_k_obs), dim=-2)
		weight = weight.reshape(weight.shape[0]//self.num_agents,self.num_agents,-1)
		# ATTENTION VALUES
		attention_values = torch.tanh(self.attention_value_layer(states))
		# print(attention_values)
		attention_values = attention_values.repeat(1,self.num_agents,1).reshape(attention_values.shape[0],self.num_agents,self.num_agents,-1)
		# SOFTMAX
		weighted_attention_values = attention_values*weight.unsqueeze(-1)
		# SOFTMAX WITH NOISE
		# weight = weight.unsqueeze(-2).repeat(1,1,self.num_agents,1).unsqueeze(-1)
		# uniform_noise = (self.noise_uniform((attention_values.view(-1).size())).reshape(attention_values.size()) - 0.5) * 0.1 #SCALING NOISE AND MAKING IT ZERO CENTRIC
		# weighted_attention_values = attention_values*weight + uniform_noise
		# SOFTMAX WITH NORMALIZATION
		# scaling_weight = F.relu(weight - self.threshold)
		# scaling_weight = torch.div(scaling_weight,torch.sum(scaling_weight,dim =-1).unsqueeze(-1))
		# ret_weight = scaling_weight
		# scaling_weight = scaling_weight.unsqueeze(-2).repeat(1,1,self.num_agents,1).unsqueeze(-1)
		# weighted_attention_values = attention_values*scaling_weight
		# print(weighted_attention_values)
		node_features = torch.sum(weighted_attention_values, dim=-2)

		Policy = F.leaky_relu(self.final_policy_layer_1(node_features))
		Policy = F.softmax(self.final_policy_layer_2(Policy), dim=-1)

		return Policy, weight




class ScalarDotProductCriticNetworkDualAttention(nn.Module):
	def __init__(self, obs_input_dim, obs_output_dim, obs_act_input_dim, obs_act_output_dim, obs_input_dim_other, obs_output_dim_other, obs_act_input_dim_other, obs_act_output_dim_other, final_input_dim, final_output_dim, num_agents, num_other, num_actions, threshold=0.1):
		super(ScalarDotProductCriticNetworkDualAttention, self).__init__()
		
		self.num_agents = num_agents
		self.num_other = num_other
		self.num_actions = num_actions
		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		# self.device = "cpu"

		self.key_layer = nn.Linear(obs_input_dim, obs_output_dim, bias=False)
		self.query_layer = nn.Linear(obs_input_dim, obs_output_dim, bias=False)
		self.attention_value_layer = nn.Linear(obs_act_input_dim, obs_act_output_dim, bias=False)
		# dimesion of key
		self.d_k_obs = obs_output_dim

		self.key_other_layer = nn.Linear(obs_input_dim_other, obs_output_dim_other, bias=False)
		self.query_other_layer = nn.Linear(obs_input_dim_other, obs_output_dim, bias=False)
		self.attention_value_other_layer = nn.Linear(obs_act_input_dim_other, obs_act_output_dim_other, bias=False)
		# dimesion of key
		self.d_k_obs_other = obs_output_dim_other



		# NOISE
		self.noise_normal = torch.distributions.Normal(loc=torch.tensor([0.0]), scale=torch.tensor([1.0]))
		self.noise_uniform = torch.rand
		# ********************************************************************************************************

		# ********************************************************************************************************
		# FCN FINAL LAYER TO GET VALUES
		self.final_value_layer_1 = nn.Linear(final_input_dim, 64, bias=False)
		self.final_value_layer_2 = nn.Linear(64, final_output_dim, bias=False)
		# ********************************************************************************************************	

		self.place_policies = torch.zeros(self.num_agents,self.num_agents,obs_act_input_dim).to(self.device)
		self.place_actions = torch.ones(self.num_agents,self.num_agents,obs_act_input_dim).to(self.device)
		one_hots = torch.ones(obs_act_input_dim)
		zero_hots = torch.zeros(obs_act_input_dim)

		for j in range(self.num_agents):
			self.place_policies[j][j] = one_hots
			self.place_actions[j][j] = zero_hots

		self.threshold = threshold
		# ********************************************************************************************************* 

		self.reset_parameters()

	def reset_parameters(self):
		"""Reinitialize learnable parameters."""
		gain_leaky = nn.init.calculate_gain('leaky_relu')

		nn.init.xavier_uniform_(self.key_layer.weight)
		nn.init.xavier_uniform_(self.query_layer.weight)
		nn.init.xavier_uniform_(self.attention_value_layer.weight)
		nn.init.xavier_uniform_(self.key_other_layer.weight)
		nn.init.xavier_uniform_(self.query_other_layer.weight)
		nn.init.xavier_uniform_(self.attention_value_other_layer.weight)


		nn.init.xavier_uniform_(self.final_value_layer_1.weight, gain=gain_leaky)
		nn.init.xavier_uniform_(self.final_value_layer_2.weight, gain=gain_leaky)



	def forward(self, states, policies, actions, states_other, actions_other):

		# input to KEY, QUERY and ATTENTION VALUE NETWORK
		obs_actions = torch.cat([states,actions],dim=-1)
		# For calculating the right advantages
		obs_policy = torch.cat([states,policies], dim=-1)

		# KEYS
		key_obs = self.key_layer(states)
		# QUERIES
		query_obs = self.query_layer(states)
		# SCORE
		score_obs = torch.bmm(query_obs,key_obs.transpose(1,2)).reshape(-1,1)
		score_obs = score_obs.reshape(-1,self.num_agents,1)
		# WEIGHT
		weight = F.softmax(score_obs/math.sqrt(self.d_k_obs), dim=-2)
		weight = weight.reshape(weight.shape[0]//self.num_agents,self.num_agents,-1)
		ret_weight = weight
		
		obs_actions = obs_actions.repeat(1,self.num_agents,1).reshape(obs_actions.shape[0],self.num_agents,self.num_agents,-1)
		obs_policy = obs_policy.repeat(1,self.num_agents,1).reshape(obs_policy.shape[0],self.num_agents,self.num_agents,-1)
		obs_actions_policies = self.place_policies*obs_policy + self.place_actions*obs_actions
		# ATTENTION VALUES
		attention_values = torch.tanh(self.attention_value_layer(obs_actions_policies))
		attention_values = attention_values.repeat(1,self.num_agents,1,1).reshape(attention_values.shape[0],self.num_agents,self.num_agents,self.num_agents,-1)

		# SOFTMAX
		weight = weight.unsqueeze(-2).repeat(1,1,self.num_agents,1).unsqueeze(-1)
		weighted_attention_values = torch.sum(attention_values*weight, dim=-2)

		# SOFTMAX WITH NOISE
		# weight = weight.unsqueeze(-2).repeat(1,1,self.num_agents,1).unsqueeze(-1)
		# uniform_noise = (self.noise_uniform((attention_values.view(-1).size())).reshape(attention_values.size()) - 0.5) * 0.1 #SCALING NOISE AND MAKING IT ZERO CENTRIC
		# weighted_attention_values = attention_values*weight + uniform_noise

		# SOFTMAX WITH NORMALIZATION
		# scaling_weight = F.relu(weight - self.threshold)
		# scaling_weight = torch.div(scaling_weight,torch.sum(scaling_weight,dim =-1).unsqueeze(-1))
		# ret_weight = scaling_weight
		# scaling_weight = scaling_weight.unsqueeze(-2).repeat(1,1,self.num_agents,1).unsqueeze(-1)
		# weighted_attention_values = attention_values*scaling_weight

		# Input to Key and Attention Value Network
		obs_actions_other = torch.cat([states_other,actions_other], dim=-1)
		# KEY
		keys_obs_other = self.key_other_layer(states_other)
		# QUERY
		query_obs_agent = self.query_other_layer(states)
		# ATTENTION VALUE
		attention_values_other = torch.tanh(self.attention_value_other_layer(obs_actions_other))
		attention_values_other = attention_values_other.repeat(1,self.num_agents,1).reshape(attention_values_other.shape[0],self.num_agents,self.num_other,-1)
		# SCORE
		score_obs_other = torch.bmm(query_obs_agent,keys_obs_other.transpose(1,2))
		# WEIGHT
		weight_other = F.softmax(score_obs_other/math.sqrt(self.d_k_obs_other), dim=-2).unsqueeze(-1)
		# WEIGHTED ATTENTION VALUES
		weighted_attention_values_other = torch.sum(attention_values_other*weight_other, dim=-2).unsqueeze(-2).repeat(1,1,self.num_agents,1)


		node_features = torch.cat([weighted_attention_values,weighted_attention_values_other], dim=-1)

		Value = F.leaky_relu(self.final_value_layer_1(node_features))
		Value = self.final_value_layer_2(Value)

		return Value, ret_weight, weight_other



class ScalarDotProductPolicyNetworkDualAttention(nn.Module):
	def __init__(self, obs_input_dim, obs_output_dim, obs_input_dim_other, obs_output_dim_other, final_input_dim, final_output_dim, num_agents, num_other, num_actions, threshold=0.1):
		super(ScalarDotProductPolicyNetworkDualAttention, self).__init__()
		
		self.num_agents = num_agents
		self.num_other = num_other
		self.num_actions = num_actions
		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		# self.device = "cpu"

		self.key_layer = nn.Linear(obs_input_dim, obs_output_dim, bias=False)
		self.query_layer = nn.Linear(obs_input_dim, obs_output_dim, bias=False)
		self.attention_value_layer = nn.Linear(obs_input_dim, obs_output_dim, bias=False)
		# dimesion of key
		self.d_k_obs = obs_output_dim

		self.key_other_layer = nn.Linear(obs_input_dim_other, obs_output_dim_other, bias=False)
		self.query_other_layer = nn.Linear(obs_input_dim, obs_output_dim, bias=False)
		self.attention_value_other_layer = nn.Linear(obs_input_dim_other, obs_output_dim_other, bias=False)
		# dimesion of key
		self.d_k_obs_other = obs_output_dim_other

		# NOISE
		self.noise_normal = torch.distributions.Normal(loc=torch.tensor([0.0]), scale=torch.tensor([1.0]))
		self.noise_uniform = torch.rand
		# ********************************************************************************************************

		# ********************************************************************************************************
		# FCN FINAL LAYER TO GET VALUES
		self.final_policy_layer_1 = nn.Linear(final_input_dim, 64, bias=False)
		self.final_policy_layer_2 = nn.Linear(64, final_output_dim, bias=False)
		# ********************************************************************************************************	

		self.threshold = threshold
		# ********************************************************************************************************* 

		self.reset_parameters()

	def reset_parameters(self):
		"""Reinitialize learnable parameters."""
		gain_leaky = nn.init.calculate_gain('leaky_relu')

		nn.init.xavier_uniform_(self.key_layer.weight)
		nn.init.xavier_uniform_(self.query_layer.weight)
		nn.init.xavier_uniform_(self.attention_value_layer.weight)
		nn.init.xavier_uniform_(self.key_other_layer.weight)
		nn.init.xavier_uniform_(self.query_other_layer.weight)
		nn.init.xavier_uniform_(self.attention_value_other_layer.weight)


		nn.init.xavier_uniform_(self.final_policy_layer_1.weight, gain=gain_leaky)
		nn.init.xavier_uniform_(self.final_policy_layer_2.weight, gain=gain_leaky)


	def forward(self, states, states_other):

		# KEYS
		key_obs = self.key_layer(states)
		# QUERIES
		query_obs = self.query_layer(states)
		# SCORE CALCULATION
		score_obs = torch.bmm(query_obs,key_obs.transpose(1,2)).reshape(-1,1)
		score_obs = score_obs.reshape(-1,self.num_agents,1)
		# WEIGHT
		weight = F.softmax(score_obs/math.sqrt(self.d_k_obs), dim=-2)
		weight = weight.reshape(weight.shape[0]//self.num_agents,self.num_agents,-1)
		# ATTENTION VALUES
		attention_values = torch.tanh(self.attention_value_layer(states))
		# print(attention_values)
		attention_values = attention_values.repeat(1,self.num_agents,1).reshape(attention_values.shape[0],self.num_agents,self.num_agents,-1)
		# SOFTMAX
		weighted_attention_values = torch.sum(attention_values*weight.unsqueeze(-1),dim=-2)
		# SOFTMAX WITH NOISE
		# weight = weight.unsqueeze(-2).repeat(1,1,self.num_agents,1).unsqueeze(-1)
		# uniform_noise = (self.noise_uniform((attention_values.view(-1).size())).reshape(attention_values.size()) - 0.5) * 0.1 #SCALING NOISE AND MAKING IT ZERO CENTRIC
		# weighted_attention_values = attention_values*weight + uniform_noise
		# SOFTMAX WITH NORMALIZATION
		# scaling_weight = F.relu(weight - self.threshold)
		# scaling_weight = torch.div(scaling_weight,torch.sum(scaling_weight,dim =-1).unsqueeze(-1))
		# ret_weight = scaling_weight
		# scaling_weight = scaling_weight.unsqueeze(-2).repeat(1,1,self.num_agents,1).unsqueeze(-1)
		# weighted_attention_values = attention_values*scaling_weight
		# print(weighted_attention_values)

		# KEYS
		key_obs_other = self.key_other_layer(states_other)
		# QUERIES
		query_obs_other = self.query_other_layer(states)
		# SCORE CALCULATION
		score_obs_other = torch.bmm(query_obs_other,key_obs_other.transpose(1,2))
		# WEIGHT
		weight_other = F.softmax(score_obs_other/math.sqrt(self.d_k_obs_other), dim=-2).unsqueeze(-1)
		# ATTENTION VALUES
		attention_values_other = torch.tanh(self.attention_value_other_layer(states_other))
		attention_values_other = attention_values_other.repeat(1,self.num_agents,1).reshape(attention_values_other.shape[0],self.num_agents,self.num_other,-1)
		# SOFTMAX
		weighted_attention_values_other = torch.sum(attention_values_other*weight_other,dim=-2)

		node_features = torch.cat([weighted_attention_values,weighted_attention_values_other], dim=-1)

		Policy = F.leaky_relu(self.final_policy_layer_1(node_features))
		Policy = F.softmax(self.final_policy_layer_2(Policy), dim=-1)

		return Policy, weight, weight_other





class ScalarDotProductCriticNetworkRevised(nn.Module):
	def __init__(self, obs_input_dim, obs_output_dim, obs_act_input_dim, obs_act_output_dim, final_input_dim, final_output_dim, num_agents, num_actions, threshold=0.1):
		super(ScalarDotProductCriticNetworkRevised, self).__init__()
		
		self.num_agents = num_agents
		self.num_actions = num_actions
		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		# self.device = "cpu"

		self.key_proc_layer = nn.Linear(obs_input_dim, obs_output_dim, bias=False)
		self.query_proc_layer = nn.Linear(obs_input_dim, obs_output_dim, bias=False)
		self.attention_value_proc_layer = nn.Linear(obs_input_dim, obs_output_dim, bias=False)
		# dimesion of key
		self.d_k_obs_proc = obs_output_dim


		self.key_layer = nn.Linear(obs_output_dim, final_input_dim, bias=False)
		self.query_layer = nn.Linear(obs_output_dim, final_input_dim, bias=False)
		self.embed_obs_act_pol = nn.Linear(obs_act_input_dim, obs_act_output_dim, bias=False)
		self.attention_value_layer = nn.Linear(obs_act_output_dim, final_input_dim, bias=False)

		# dimesion of key
		self.d_k_obs = final_input_dim

		# NOISE
		self.noise_normal = torch.distributions.Normal(loc=torch.tensor([0.0]), scale=torch.tensor([1.0]))
		self.noise_uniform = torch.rand
		# ********************************************************************************************************

		# ********************************************************************************************************
		# FCN FINAL LAYER TO GET VALUES
		self.final_value_layer_1 = nn.Linear(final_input_dim, 256, bias=False)
		self.final_value_layer_2 = nn.Linear(256, final_output_dim, bias=False)
		# ********************************************************************************************************	

		self.place_policies = torch.zeros(self.num_agents,self.num_agents,obs_act_input_dim).to(self.device)
		self.place_actions = torch.ones(self.num_agents,self.num_agents,obs_act_input_dim).to(self.device)
		one_hots = torch.ones(obs_act_input_dim)
		zero_hots = torch.zeros(obs_act_input_dim)

		for j in range(self.num_agents):
			self.place_policies[j][j] = one_hots
			self.place_actions[j][j] = zero_hots

		self.threshold = threshold
		self.obs_act_input_dim = obs_act_input_dim
		# ********************************************************************************************************* 

		self.reset_parameters()


	def mixing_actions_policies(self):
		self.place_policies = torch.zeros(self.num_agents,self.num_agents,self.obs_act_input_dim).to(self.device)
		self.place_actions = torch.ones(self.num_agents,self.num_agents,self.obs_act_input_dim).to(self.device)
		one_hots = torch.ones(self.obs_act_input_dim)
		zero_hots = torch.zeros(self.obs_act_input_dim)

		for j in range(self.num_agents):
			self.place_policies[j][j] = one_hots
			self.place_actions[j][j] = zero_hots


	def reset_parameters(self):
		"""Reinitialize learnable parameters."""
		gain_leaky = nn.init.calculate_gain('leaky_relu')

		nn.init.xavier_uniform_(self.key_proc_layer.weight)
		nn.init.xavier_uniform_(self.query_proc_layer.weight)
		nn.init.xavier_uniform_(self.attention_value_proc_layer.weight)

		nn.init.xavier_uniform_(self.key_layer.weight)
		nn.init.xavier_uniform_(self.query_layer.weight)
		nn.init.xavier_uniform_(self.embed_obs_act_pol.weight)
		nn.init.xavier_uniform_(self.attention_value_layer.weight)


		nn.init.xavier_uniform_(self.final_value_layer_1.weight, gain=gain_leaky)
		nn.init.xavier_uniform_(self.final_value_layer_2.weight, gain=gain_leaky)



	def forward(self, states, policies, actions):

		# KEYS
		key_obs_proc = self.key_proc_layer(states)
		print("key_obs_proc")
		print(key_obs_proc)
		# QUERIES
		query_obs_proc = self.query_proc_layer(states)
		print("query_obs_proc")
		print(query_obs_proc)
		# ATTENTION VALUES
		attention_values_proc = self.attention_value_proc_layer(states)
		print("attention_values_proc")
		print(attention_values_proc)
		attention_values_proc = attention_values_proc.repeat(1,self.num_agents,1).reshape(attention_values_proc.shape[0],self.num_agents,self.num_agents,-1)
		# WEIGHTS
		score_obs_proc = torch.bmm(query_obs_proc,key_obs_proc.transpose(1,2)).reshape(-1,1)
		score_obs_proc = score_obs_proc.reshape(-1,self.num_agents,1)
		weight_1 = F.softmax(score_obs_proc/math.sqrt(self.d_k_obs_proc), dim=-2)
		print("weight_1")
		print(weight_1)
		weight_1 = weight_1.reshape(weight_1.shape[0]//self.num_agents,self.num_agents,self.num_agents,1)
		weighted_attention_value_proc = torch.sum(weight_1*attention_values_proc, dim=-2)
		print("weighted_attention_value_proc")
		print(weighted_attention_value_proc)

		# KEYS
		keys_obs_embed = self.key_layer(weighted_attention_value_proc)
		print("keys_obs_embed")
		print(keys_obs_embed)
		# QUERIES
		query_obs_embed = self.query_layer(weighted_attention_value_proc)
		print("query_obs_embed")
		print(keys_obs_embed)
		# WEIGHTS
		score_obs_embed = torch.bmm(query_obs_embed,keys_obs_embed.transpose(1,2)).reshape(-1,1)
		score_obs_embed = score_obs_embed.reshape(-1,self.num_agents,1)
		weight_2 = F.softmax(score_obs_embed/math.sqrt(self.d_k_obs), dim=-2)
		weight_2 = weight_2.reshape(weight_2.shape[0]//self.num_agents,self.num_agents,-1)
		ret_weight = weight_2
		print("weight_2")
		print(weight_2)
		
		obs_actions = torch.cat([states,actions],dim=-1)
		obs_policy = torch.cat([states,policies], dim=-1)
		obs_actions = obs_actions.repeat(1,self.num_agents,1).reshape(obs_actions.shape[0],self.num_agents,self.num_agents,-1)
		obs_policy = obs_policy.repeat(1,self.num_agents,1).reshape(obs_policy.shape[0],self.num_agents,self.num_agents,-1)
		# RANDOMIZING NUMBER OF AGENTS
		# self.mixing_actions_policies()
		obs_actions_policies = self.place_policies*obs_policy + self.place_actions*obs_actions

		# embedding the observation_actions_policies
		obs_actions_policies_embed = self.embed_obs_act_pol(obs_actions_policies)
		print("obs_actions_policies_embed")
		print(obs_actions_policies_embed)
		attention_values = self.attention_value_layer(obs_actions_policies_embed)
		print("attention_values")
		print(attention_values)
		attention_values = attention_values.repeat(1,self.num_agents,1,1).reshape(attention_values.shape[0],self.num_agents,self.num_agents,self.num_agents,-1)

		# SOFTMAX
		weight_2 = weight_2.unsqueeze(-2).repeat(1,1,self.num_agents,1).unsqueeze(-1)
		weighted_attention_values = attention_values*weight_2

		# SOFTMAX WITH NOISE
		# weight = weight.unsqueeze(-2).repeat(1,1,self.num_agents,1).unsqueeze(-1)
		# uniform_noise = (self.noise_uniform((attention_values.view(-1).size())).reshape(attention_values.size()) - 0.5) * 0.1 #SCALING NOISE AND MAKING IT ZERO CENTRIC
		# weighted_attention_values = attention_values*weight + uniform_noise

		# SOFTMAX WITH NORMALIZATION
		# scaling_weight = F.relu(weight - self.threshold)
		# scaling_weight = torch.div(scaling_weight,torch.sum(scaling_weight,dim =-1).unsqueeze(-1))
		# ret_weight = scaling_weight
		# scaling_weight = scaling_weight.unsqueeze(-2).repeat(1,1,self.num_agents,1).unsqueeze(-1)
		# weighted_attention_values = attention_values*scaling_weight

		node_features = torch.sum(weighted_attention_values, dim=-2)
		print("node_features")
		print(node_features)

		Value = F.leaky_relu(self.final_value_layer_1(node_features))
		Value = self.final_value_layer_2(Value)
		print("Value")
		print(Value)

		return Value, weight_1, ret_weight
		


class ScalarDotProductPolicyNetworkRevised(nn.Module):
	def __init__(self, obs_input_dim, obs_output_dim, final_input_dim, final_output_dim, num_agents, num_actions, threshold=0.1):
		super(ScalarDotProductPolicyNetworkRevised, self).__init__()
		
		self.num_agents = num_agents
		self.num_actions = num_actions
		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		# self.device = "cpu"

		self.key_proc_layer = nn.Linear(obs_input_dim, obs_output_dim, bias=False)
		self.query_proc_layer = nn.Linear(obs_input_dim, obs_output_dim, bias=False)
		self.attention_value_proc_layer = nn.Linear(obs_input_dim, obs_output_dim, bias=False)
		# dimesion of key
		self.d_k_obs_proc = obs_output_dim


		self.key_layer = nn.Linear(obs_output_dim, final_input_dim, bias=False)
		self.query_layer = nn.Linear(obs_output_dim, final_input_dim, bias=False)
		self.attention_value_layer = nn.Linear(obs_output_dim, final_input_dim, bias=False)

		# dimesion of key
		self.d_k_obs = final_input_dim


		# NOISE
		self.noise_normal = torch.distributions.Normal(loc=torch.tensor([0.0]), scale=torch.tensor([1.0]))
		self.noise_uniform = torch.rand
		# ********************************************************************************************************

		# ********************************************************************************************************
		# FCN FINAL LAYER TO GET VALUES
		self.final_policy_layer_1 = nn.Linear(final_input_dim, 256, bias=False)
		self.final_policy_layer_2 = nn.Linear(256, final_output_dim, bias=False)
		# ********************************************************************************************************	

		self.threshold = threshold
		# ********************************************************************************************************* 

		self.reset_parameters()

	def reset_parameters(self):
		"""Reinitialize learnable parameters."""
		gain_leaky = nn.init.calculate_gain('leaky_relu')

		nn.init.xavier_uniform_(self.key_proc_layer.weight)
		nn.init.xavier_uniform_(self.query_proc_layer.weight)
		nn.init.xavier_uniform_(self.attention_value_proc_layer.weight)

		nn.init.xavier_uniform_(self.key_layer.weight)
		nn.init.xavier_uniform_(self.query_layer.weight)
		nn.init.xavier_uniform_(self.attention_value_layer.weight)


		nn.init.xavier_uniform_(self.final_policy_layer_1.weight, gain=gain_leaky)
		nn.init.xavier_uniform_(self.final_policy_layer_2.weight, gain=gain_leaky)



	def forward(self, states):

		# KEYS
		key_obs_proc = self.key_proc_layer(states)
		# QUERIES
		query_obs_proc = self.query_proc_layer(states)
		# ATTENTION VALUES
		attention_values_proc = self.attention_value_proc_layer(states)
		attention_values_proc = attention_values_proc.repeat(1,self.num_agents,1).reshape(attention_values_proc.shape[0],self.num_agents,self.num_agents,-1)
		# WEIGHTS
		score_obs_proc = torch.bmm(query_obs_proc,key_obs_proc.transpose(1,2)).reshape(-1,1)
		score_obs_proc = score_obs_proc.reshape(-1,self.num_agents,1)
		weight_1 = F.softmax(score_obs_proc/math.sqrt(self.d_k_obs_proc), dim=-2)
		weight_1 = weight_1.reshape(weight_1.shape[0]//self.num_agents,self.num_agents,self.num_agents,1)
		weighted_attention_value_proc = torch.sum(weight_1*attention_values_proc, dim=-2)

		# KEYS
		key_obs = self.key_layer(weighted_attention_value_proc)
		# QUERIES
		query_obs = self.query_layer(weighted_attention_value_proc)
		# SCORE CALCULATION
		score_obs = torch.bmm(query_obs,key_obs.transpose(1,2)).reshape(-1,1)
		score_obs = score_obs.reshape(-1,self.num_agents,1)
		# WEIGHT
		weight = F.softmax(score_obs/math.sqrt(self.d_k_obs), dim=-2)
		weight = weight.reshape(weight.shape[0]//self.num_agents,self.num_agents,self.num_agents,1)
		# ATTENTION VALUES
		attention_values = self.attention_value_layer(weighted_attention_value_proc)
		# print(attention_values)
		attention_values = attention_values.repeat(1,self.num_agents,1).reshape(attention_values.shape[0],self.num_agents,self.num_agents,-1)
		# SOFTMAX
		weighted_attention_values = attention_values*weight
		# SOFTMAX WITH NOISE
		# weight = weight.unsqueeze(-2).repeat(1,1,self.num_agents,1).unsqueeze(-1)
		# uniform_noise = (self.noise_uniform((attention_values.view(-1).size())).reshape(attention_values.size()) - 0.5) * 0.1 #SCALING NOISE AND MAKING IT ZERO CENTRIC
		# weighted_attention_values = attention_values*weight + uniform_noise
		# SOFTMAX WITH NORMALIZATION
		# scaling_weight = F.relu(weight - self.threshold)
		# scaling_weight = torch.div(scaling_weight,torch.sum(scaling_weight,dim =-1).unsqueeze(-1))
		# ret_weight = scaling_weight
		# scaling_weight = scaling_weight.unsqueeze(-2).repeat(1,1,self.num_agents,1).unsqueeze(-1)
		# weighted_attention_values = attention_values*scaling_weight
		node_features = torch.sum(weighted_attention_values, dim=-2)

		Policy = F.leaky_relu(self.final_policy_layer_1(node_features))
		Policy = F.softmax(self.final_policy_layer_2(Policy), dim=-1)

		return Policy, weight_1, weight
