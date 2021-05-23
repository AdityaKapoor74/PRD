from typing import Any, List, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import datetime
import math




class ScalarDotProductCriticNetwork(nn.Module):
	def __init__(self, obs_act_input_dim, obs_act_output_dim, obs_act_input_dim_people, obs_act_output_dim_people, final_input_dim, final_output_dim, num_agents, num_people, num_actions, threshold=0.1):
		super(ScalarDotProductCriticNetwork, self).__init__()
		
		self.num_agents = num_agents
		self.num_people = num_people
		self.num_actions = num_actions
		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		# self.device = "cpu"

		self.key_layer = nn.Linear(obs_act_input_dim, obs_act_output_dim, bias=False)
		self.query_layer = nn.Linear(obs_act_input_dim, obs_act_output_dim, bias=False)
		self.attention_value_layer = nn.Linear(obs_act_input_dim, obs_act_output_dim, bias=False)
		# dimesion of key
		self.d_k_obs_act = obs_act_output_dim

		self.key_people_layer = nn.Linear(obs_act_input_dim_people, obs_act_output_dim_people, bias=False)
		self.query_people_layer = nn.Linear(obs_act_input_dim, obs_act_output_dim, bias=False)
		self.attention_value_people_layer = nn.Linear(obs_act_input_dim_people, obs_act_output_dim_people, bias=False)
		# dimesion of key
		self.d_k_obs_act_people = obs_act_output_dim_people



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
		gain = nn.init.calculate_gain('leaky_relu')

		nn.init.xavier_uniform_(self.key_layer.weight)
		nn.init.xavier_uniform_(self.query_layer.weight)
		nn.init.xavier_uniform_(self.attention_value_layer.weight)

		nn.init.xavier_uniform_(self.key_people_layer.weight)
		nn.init.xavier_uniform_(self.query_people_layer.weight)
		nn.init.xavier_uniform_(self.attention_value_people_layer.weight)


		nn.init.xavier_uniform_(self.final_value_layer_1.weight, gain=gain)
		nn.init.xavier_uniform_(self.final_value_layer_2.weight, gain=gain)



	def forward(self, states, policies, actions, states_people, actions_people):

		# input to KEY, QUERY and ATTENTION VALUE NETWORK
		obs_actions = torch.cat([states,actions],dim=-1)
		# For calculating the right advantages
		obs_policy = torch.cat([states,policies], dim=-1)
		# KEYS
		key_obs_actions = self.key_layer(obs_actions)
		# QUERIES
		query_obs_actions = self.query_layer(obs_actions)
		# SCORE
		score_obs_actions = torch.bmm(query_obs_actions,key_obs_actions.transpose(1,2)).transpose(1,2).reshape(-1,1)
		score_obs_actions = score_obs_actions.reshape(-1,self.num_agents,1)
		# WEIGHT
		weight = F.softmax(score_obs_actions/math.sqrt(self.d_k_obs_act), dim=-2)
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
		weighted_attention_values = torch.mean(attention_values*weight, dim=-2)

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

		# Input to Query Net
		obs_actions = torch.cat([states,actions],dim=-1)
		# Input to Key and Attention Value Network
		obs_actions_people = torch.cat([states_people,actions_people], dim=-1)
		# KEY
		keys_obs_actions_people = self.key_people_layer(obs_actions_people)
		# QUERY
		query_obs_actions_agent = self.query_people_layer(obs_actions)
		# ATTENTION VALUE
		attention_values_people = torch.tanh(self.attention_value_people_layer(obs_actions_people))
		attention_values_people = attention_values_people.repeat(1,self.num_agents,1).reshape(attention_values_people.shape[0],self.num_agents,self.num_people,-1)
		# SCORE
		score_obs_actions_people = torch.bmm(query_obs_actions_agent,keys_obs_actions_people.transpose(1,2))
		# WEIGHT
		weight_people = F.softmax(score_obs_actions_people/math.sqrt(self.d_k_obs_act_people), dim=-2).unsqueeze(-1)
		# WEIGHTED ATTENTION VALUES
		weighted_attention_values_people = torch.mean(attention_values_people*weight_people, dim=-2).unsqueeze(-2).repeat(1,1,self.num_agents,1)


		node_features = torch.cat([weighted_attention_values,weighted_attention_values_people], dim=-1)

		Value = F.leaky_relu(self.final_value_layer_1(node_features))
		Value = self.final_value_layer_2(Value)

		return Value, ret_weight, weight_people



class ScalarDotProductPolicyNetwork(nn.Module):
	def __init__(self, obs_input_dim, obs_output_dim, obs_input_people_dim, obs_output_people_dim, final_input_dim, final_output_dim, num_agents, num_people, num_actions, threshold=0.1):
		super(ScalarDotProductPolicyNetwork, self).__init__()
		
		self.num_agents = num_agents
		self.num_people = num_people
		self.num_actions = num_actions
		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		# self.device = "cpu"

		self.key_layer = nn.Linear(obs_input_dim, obs_output_dim, bias=False)
		self.query_layer = nn.Linear(obs_input_dim, obs_output_dim, bias=False)
		self.attention_value_layer = nn.Linear(obs_input_dim, obs_output_dim, bias=False)
		# dimesion of key
		self.d_k_obs = obs_output_dim

		self.key_people_layer = nn.Linear(obs_input_people_dim, obs_output_people_dim, bias=False)
		self.query_people_layer = nn.Linear(obs_input_dim, obs_output_dim, bias=False)
		self.attention_value_people_layer = nn.Linear(obs_input_people_dim, obs_output_people_dim, bias=False)
		# dimesion of key
		self.d_k_obs_people = obs_output_people_dim

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

		nn.init.xavier_uniform_(self.key_people_layer.weight)
		nn.init.xavier_uniform_(self.query_people_layer.weight)
		nn.init.xavier_uniform_(self.attention_value_people_layer.weight)


		nn.init.xavier_uniform_(self.final_policy_layer_1.weight, gain=gain)
		nn.init.xavier_uniform_(self.final_policy_layer_2.weight, gain=gain)



	def forward(self, states, states_people):

		# KEYS
		key_obs = self.key_layer(states)
		# QUERIES
		query_obs = self.query_layer(states)
		# SCORE CALCULATION
		score_obs = torch.bmm(query_obs,key_obs.transpose(1,2)).transpose(1,2).reshape(-1,1)
		score_obs = score_obs.reshape(-1,self.num_agents,1)
		# WEIGHT
		weight = F.softmax(score_obs/math.sqrt(self.d_k_obs), dim=-2)
		weight = weight.reshape(weight.shape[0]//self.num_agents,self.num_agents,-1)
		# ATTENTION VALUES
		attention_values = torch.tanh(self.attention_value_layer(states))
		# print(attention_values)
		attention_values = attention_values.repeat(1,self.num_agents,1).reshape(attention_values.shape[0],self.num_agents,self.num_agents,-1)
		# SOFTMAX
		weighted_attention_values = torch.mean(attention_values*weight.unsqueeze(-1),dim=-2)
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
		key_obs_people = self.key_people_layer(states_people)
		# QUERIES
		query_obs_people = self.query_people_layer(states)
		# SCORE CALCULATION
		score_obs_people = torch.bmm(query_obs_people,key_obs_people.transpose(1,2))
		# WEIGHT
		weight_people = F.softmax(score_obs_people/math.sqrt(self.d_k_obs_people), dim=-2).unsqueeze(-1)
		# ATTENTION VALUES
		attention_values_people = torch.tanh(self.attention_value_people_layer(states_people))
		attention_values_people = attention_values_people.repeat(1,self.num_agents,1).reshape(attention_values_people.shape[0],self.num_agents,self.num_people,-1)
		# SOFTMAX
		weighted_attention_values_people = torch.mean(attention_values_people*weight_people,dim=-2)

		node_features = torch.cat([weighted_attention_values,weighted_attention_values_people], dim=-1)

		Policy = F.leaky_relu(self.final_policy_layer_1(node_features))
		Policy = F.softmax(self.final_policy_layer_2(Policy), dim=-1)

		return Policy, weight, weight_people