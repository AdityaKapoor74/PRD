from typing import Any, List, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import datetime
import math

# *******************************************
# Q(s,a) 
# *******************************************

def create_model(
	layer_sizes: Tuple,
	weight_init: str = "xavier_uniform",
	activation_func: str = "leaky_relu"
	):

	layers = []
	limit = len(layer_sizes)

	# add more activations
	if activation_func == "tanh":
		activation = nn.Tanh()
	elif activation_func == "relu":
		activation = nn.ReLU()
	elif activation_func == "leaky_relu":
		activation = nn.LeakyReLU()

	# add more weight init
	if weight_init == "xavier_uniform":
		weight_init = torch.nn.init.xavier_uniform_
	elif weight_init == "xavier_normal":
		weight_init = torch.nn.init.xavier_normal_

	for layer in range(limit - 1):
		act = activation if layer < limit - 2 else nn.Identity()
		layers += [nn.Linear(layer_sizes[layer], layer_sizes[layer + 1])]
		weight_init(layers[-1].weight)
		layers += [act]

	return nn.Sequential(*layers)




class ScalarDotProductCriticNetwork(nn.Module):
	def __init__(self, obs_act_input_dim, obs_act_output_dim, final_input_dim, final_output_dim, num_agents, num_actions, threshold=0.1):
		super(ScalarDotProductCriticNetwork, self).__init__()
		
		self.num_agents = num_agents
		self.num_actions = num_actions
		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		# self.device = "cpu"

		self.key_layer = nn.Linear(obs_act_input_dim, obs_act_output_dim, bias=False)

		self.query_layer = nn.Linear(obs_act_input_dim, obs_act_output_dim, bias=False)

		self.attention_value_layer = nn.Linear(obs_act_input_dim, obs_act_output_dim, bias=False)

		# dimesion of key
		self.d_k_obs_act = obs_act_output_dim

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


		nn.init.xavier_uniform_(self.final_value_layer_1.weight, gain=gain)
		nn.init.xavier_uniform_(self.final_value_layer_2.weight, gain=gain)



	def forward(self, states, policies, actions):

		# input to KEY, QUERY and ATTENTION VALUE NETWORK
		obs_actions = torch.cat([states,actions],dim=-1)
		# print("OBSERVATIONS ACTIONS")
		# print(obs_actions)

		# For calculating the right advantages
		obs_policy = torch.cat([states,policies], dim=-1)

		# KEYS
		key_obs_actions = self.key_layer(obs_actions)

		# QUERIES
		query_obs_actions = self.query_layer(obs_actions)

		score_obs_actions = torch.bmm(query_obs_actions,key_obs_actions.transpose(1,2)).transpose(1,2).reshape(-1,1)

		score_obs_actions = score_obs_actions.reshape(-1,self.num_agents,1)


		weight = F.softmax(score_obs_actions/math.sqrt(self.d_k_obs_act), dim=-2)
		weight = weight.reshape(weight.shape[0]//self.num_agents,self.num_agents,-1)
		ret_weight = weight
		
		obs_actions = obs_actions.repeat(1,self.num_agents,1).reshape(obs_actions.shape[0],self.num_agents,self.num_agents,-1)

		obs_policy = obs_policy.repeat(1,self.num_agents,1).reshape(obs_policy.shape[0],self.num_agents,self.num_agents,-1)

		obs_actions_policies = self.place_policies*obs_policy + self.place_actions*obs_actions

		attention_values = torch.tanh(self.attention_value_layer(obs_actions_policies))

		current_node_states = states.unsqueeze(-2).repeat(1,1,self.num_agents,1)

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


		# node_features = torch.cat([current_node_states,torch.mean(weighted_attention_values, dim=-2)], dim=-1)
		node_features = torch.mean(weighted_attention_values, dim=-2)

		Value = F.leaky_relu(self.final_value_layer_1(node_features))
		Value = self.final_value_layer_2(Value)

		return Value, ret_weight



class GraphAttentionCriticNetwork(nn.Module):
	def __init__(self, obs_act_input_dim, obs_act_output_dim, final_input_dim, final_output_dim, num_agents, num_actions, threshold=0.1):
		super(GraphAttentionCriticNetwork, self).__init__()
		
		self.num_agents = num_agents
		self.num_actions = num_actions
		# self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		self.device = "cpu"

		self.embedding = nn.Linear(obs_act_input_dim, obs_act_output_dim, bias=False)

		self.attention = nn.Linear(2*obs_act_output_dim, 1, bias=False)

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

		nn.init.xavier_uniform_(self.embedding.weight)
		nn.init.xavier_uniform_(self.attention.weight)


		nn.init.xavier_uniform_(self.final_value_layer_1.weight, gain=gain)
		nn.init.xavier_uniform_(self.final_value_layer_2.weight, gain=gain)



	def forward(self, states, policies, actions):

		# input to KEY, QUERY and ATTENTION VALUE NETWORK
		obs_actions = torch.cat([states,actions],dim=-1)

		# For calculating the right advantages
		obs_policy = torch.cat([states,policies], dim=-1)

		embeddings = self.embedding(obs_actions)

		# print("EMBEDDINGS")
		# print(embeddings)
		source_nodes = embeddings.unsqueeze(-2).repeat(1,1,self.num_agents,1)
		# print("SOURCE NODES")
		# print(source_nodes)
		neighboring_nodes = embeddings.unsqueeze(-3).repeat(1,self.num_agents,1,1)
		# print("NEIGHBORING NODE")
		# print(neighboring_nodes)

		attention_input = torch.cat([source_nodes,neighboring_nodes], dim=-1)
		attention_scores = F.leaky_relu(self.attention(attention_input))
		weight = F.softmax(attention_scores, dim=-2)
		ret_weight = weight
		weight = weight.repeat(1,1,self.num_agents,1).reshape(-1,self.num_agents,self.num_agents,self.num_agents,1)
		# print("WEIGHT")
		# print(weight)


		obs_actions = obs_actions.repeat(1,self.num_agents,1).reshape(obs_actions.shape[0],self.num_agents,self.num_agents,-1)
		obs_policy = obs_policy.repeat(1,self.num_agents,1).reshape(obs_policy.shape[0],self.num_agents,self.num_agents,-1)
		obs_actions_policies = self.place_policies*obs_policy + self.place_actions*obs_actions
		embeddings = torch.tanh(self.embedding(obs_actions_policies))

		current_node_states = states.unsqueeze(-2).repeat(1,1,self.num_agents,1)

		embeddings = embeddings.repeat(1,self.num_agents,1,1).reshape(embeddings.shape[0],self.num_agents,self.num_agents,self.num_agents,-1)

		# SOFTMAX
		weighted_embeddings = embeddings*weight

		# SOFTMAX WITH NOISE
		# uniform_noise = (self.noise_uniform((attention_values.view(-1).size())).reshape(attention_values.size()) - 0.5) * 0.1 #SCALING NOISE AND MAKING IT ZERO CENTRIC
		# weighted_embeddings = embeddings*weight + uniform_noise

		# SOFTMAX WITH NORMALIZATION
		# scaling_weight = F.relu(weight - self.threshold)
		# scaling_weight = torch.div(scaling_weight,torch.sum(scaling_weight,dim =-2).repeat(1,1,1,self.num_agents).unsqueeze(-1))
		# weighted_embeddings = weighted_embeddings*scaling_weight


		node_features = torch.cat([current_node_states,torch.mean(weighted_embeddings, dim=-2)], dim=-1)

		Value = F.leaky_relu(self.final_value_layer_1(node_features))
		Value = self.final_value_layer_2(Value)

		return Value, ret_weight



class QNetwork(nn.Module):
	def __init__(self, obs_input_dim, obs_output_dim, obs_act_input_dim, obs_act_output_dim, final_input_dim, final_output_dim, num_agents, num_actions):
		super(QNetwork, self).__init__()
		
		self.num_agents = num_agents
		self.num_actions = num_actions
		# self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		self.device = "cpu"



		# OBSERVATIONS
		self.key_layer_obs = nn.Linear(obs_input_dim, obs_output_dim, bias=False)

		self.query_layer_obs = nn.Linear(obs_input_dim, obs_output_dim, bias=False)

		self.attention_value_layer_obs = nn.Linear(obs_input_dim, obs_output_dim, bias=False)
		# dimesion of key
		self.d_k_obs = obs_output_dim



		# OBSERVATIONS AND ACTIONS
		self.key_layer_obs_act = nn.Linear(obs_act_input_dim, obs_act_output_dim, bias=False)

		self.query_layer_obs_act = nn.Linear(obs_act_input_dim, obs_act_output_dim, bias=False)

		self.attention_value_layer_obs_act = nn.Linear(obs_act_input_dim, obs_act_output_dim, bias=False)
		# dimesion of key
		self.d_k_obs_act = obs_act_output_dim



		# ********************************************************************************************************
		# FCN FINAL LAYER TO GET VALUES
		self.final_value_layer_1 = nn.Linear(final_input_dim, 64, bias=False)
		self.final_value_layer_2 = nn.Linear(64, final_output_dim, bias=False)
		# ********************************************************************************************************	

		self.reset_parameters()

	def reset_parameters(self):
		"""Reinitialize learnable parameters."""
		gain = nn.init.calculate_gain('leaky_relu')

		nn.init.xavier_uniform_(self.key_layer_obs.weight)
		nn.init.xavier_uniform_(self.query_layer_obs.weight)
		nn.init.xavier_uniform_(self.attention_value_layer_obs.weight)

		nn.init.xavier_uniform_(self.key_layer_obs_act.weight)
		nn.init.xavier_uniform_(self.query_layer_obs_act.weight)
		nn.init.xavier_uniform_(self.attention_value_layer_obs_act.weight)


		nn.init.xavier_uniform_(self.final_value_layer_1.weight, gain=gain)
		nn.init.xavier_uniform_(self.final_value_layer_2.weight, gain=gain)



	def forward(self, states, actions):

		# SCALAR DOT PRODUCT FOR OBSERVATION

		# KEYS
		key_obs = self.key_layer_obs(states)
		# QUERIES
		query_obs = self.query_layer_obs(states)
		# ATTENTION VALUES
		attention_values_obs = self.attention_value_layer_obs(states).unsqueeze(1).repeat(1,1,self.num_agents,1).reshape(states.shape[0],self.num_agents,self.num_agents,-1)
		# SCORE
		score_obs = torch.bmm(query_obs,key_obs.transpose(1,2)).transpose(1,2).reshape(-1,1)
		score_obs = score_obs.reshape(-1,self.num_agents,1)
		# WEIGHT
		weight_obs = F.softmax(score_obs/math.sqrt(self.d_k_obs), dim=-2)
		weight_obs = weight_obs.reshape(weight_obs.shape[0]//self.num_agents,self.num_agents,-1).unsqueeze(-1)
		# FINAL OBSERVATIONS
		final_observations = torch.mean(weight_obs*attention_values_obs, dim=-2)

		# SCALAR DOT PRODUCT FOR OBSERVATION AND ACTIONS

		obs_actions = torch.cat([states,actions],dim=-1)
		# KEYS
		key_obs_actions = self.key_layer_obs_act(obs_actions)
		# QUERIES
		query_obs_actions = self.query_layer_obs_act(obs_actions)
		# SCORE
		score_obs_actions = torch.bmm(query_obs_actions,key_obs_actions.transpose(1,2)).transpose(1,2).reshape(-1,1)
		score_obs_actions = score_obs_actions.reshape(-1,self.num_agents,1)
		# WEIGHT
		weight_obs_actions = F.softmax(score_obs_actions/math.sqrt(self.d_k_obs_act), dim=-2)
		weight_obs_actions = weight_obs_actions.reshape(weight_obs_actions.shape[0]//self.num_agents,self.num_agents,-1).unsqueeze(-1)
		# ATTENTION VALUES
		attention_values_obs_act = self.attention_value_layer_obs_act(obs_actions).unsqueeze(1).repeat(1,1,self.num_agents,1).reshape(obs_actions.shape[0],self.num_agents,self.num_agents,-1)
		# FINAL OBSERVATION AND ACTIONS
		final_observations_actions = torch.mean(weight_obs_actions*attention_values_obs_act, dim=-2)

		# NODE FEATURES
		node_features = torch.cat([final_observations,final_observations_actions], dim=-1)

		Q_value = F.leaky_relu(self.final_value_layer_1(node_features))
		Q_value = self.final_value_layer_2(Q_value)

		return Q_value, weight_obs, weight_obs_actions



class DualAttentionCriticNetwork(nn.Module):
	def __init__(self, obs_input_dim, obs_output_dim, obs_act_input_dim, obs_act_output_dim, final_input_dim, final_output_dim, num_agents, num_actions, threshold=0.1):
		super(DualAttentionCriticNetwork, self).__init__()
		
		self.num_agents = num_agents
		self.num_actions = num_actions
		# self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		self.device = "cpu"

		# OBSERVATIONS
		self.key_layer_obs = nn.Linear(obs_input_dim, obs_output_dim, bias=False)

		self.query_layer_obs = nn.Linear(obs_input_dim, obs_output_dim, bias=False)

		self.attention_value_layer_obs = nn.Linear(obs_input_dim, obs_output_dim, bias=False)
		# dimesion of key
		self.d_k_obs = obs_output_dim



		# OBSERVATIONS AND ACTIONS
		self.key_layer_obs_act = nn.Linear(obs_act_input_dim, obs_act_output_dim, bias=False)

		self.query_layer_obs_act = nn.Linear(obs_act_input_dim, obs_act_output_dim, bias=False)

		self.attention_value_layer_obs_act = nn.Linear(obs_act_input_dim, obs_act_output_dim, bias=False)
		# dimesion of key
		self.d_k_obs_act = obs_act_output_dim


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

		nn.init.xavier_uniform_(self.key_layer_obs.weight)
		nn.init.xavier_uniform_(self.query_layer_obs.weight)
		nn.init.xavier_uniform_(self.attention_value_layer_obs.weight)

		nn.init.xavier_uniform_(self.key_layer_obs_act.weight)
		nn.init.xavier_uniform_(self.query_layer_obs_act.weight)
		nn.init.xavier_uniform_(self.attention_value_layer_obs_act.weight)


		nn.init.xavier_uniform_(self.final_value_layer_1.weight, gain=gain)
		nn.init.xavier_uniform_(self.final_value_layer_2.weight, gain=gain)



	def forward(self, states, policies, actions):

		# ATTENTION NETWORK FOR OBSERVATIONS AND ACTIONS
		# input to KEY, QUERY and ATTENTION VALUE NETWORK
		obs_actions = torch.cat([states,actions],dim=-1)
		# For calculating the right advantages
		obs_policy = torch.cat([states,policies], dim=-1)
		# KEYS
		key_obs_actions = self.key_layer_obs_act(obs_actions)
		# QUERIES
		query_obs_actions = self.query_layer_obs_act(obs_actions)
		# SCORE
		score_obs_actions = torch.bmm(query_obs_actions,key_obs_actions.transpose(1,2)).transpose(1,2).reshape(-1,1)
		score_obs_actions = score_obs_actions.reshape(-1,self.num_agents,1)
		# WEIGHT
		weight_obs_actions = F.softmax(score_obs_actions/math.sqrt(self.d_k_obs_act), dim=-2)
		weight_obs_actions = weight_obs_actions.reshape(weight_obs_actions.shape[0]//self.num_agents,self.num_agents,-1)
		ret_weight_obs_actions = weight_obs_actions
		# MERGING OBSERVATION AND ACTIONS
		obs_actions = obs_actions.repeat(1,self.num_agents,1).reshape(obs_actions.shape[0],self.num_agents,self.num_agents,-1)
		obs_policy = obs_policy.repeat(1,self.num_agents,1).reshape(obs_policy.shape[0],self.num_agents,self.num_agents,-1)
		obs_actions_policies = self.place_policies*obs_policy + self.place_actions*obs_actions
		# ATTENTION VALUES
		attention_values = self.attention_value_layer_obs_act(obs_actions_policies)
		attention_values = attention_values.repeat(1,self.num_agents,1,1).reshape(attention_values.shape[0],self.num_agents,self.num_agents,self.num_agents,-1)
		# WEIGHTED AVERAGE OF ATTENTION VALUES
		weight_obs_actions = weight_obs_actions.unsqueeze(-2).repeat(1,1,self.num_agents,1).unsqueeze(-1)
		uniform_noise = (self.noise_uniform((attention_values.view(-1).size())).reshape(attention_values.size()) - 0.5) * 0.1 #SCALING NOISE AND MAKING IT ZERO CENTRIC
		weighted_attention_values = torch.mean(attention_values*weight_obs_actions + uniform_noise, dim=-2)



		# ATTENTION NETWORK FOR OBSERVATION
		# KEYS
		key_obs = self.key_layer_obs(states)
		# QUERIES
		query_obs = self.query_layer_obs(states)
		# ATTENTION VALUES
		attention_values_obs = self.attention_value_layer_obs(states).unsqueeze(1).repeat(1,1,self.num_agents,1).reshape(states.shape[0],self.num_agents,self.num_agents,-1)
		# SCORE
		score_obs = torch.bmm(query_obs,key_obs.transpose(1,2)).transpose(1,2).reshape(-1,1)
		score_obs = score_obs.reshape(-1,self.num_agents,1)
		# WEIGHT
		weight_obs = F.softmax(score_obs/math.sqrt(self.d_k_obs), dim=-2)
		weight_obs = weight_obs.reshape(weight_obs.shape[0]//self.num_agents,self.num_agents,-1).unsqueeze(-1)
		# FINAL OBSERVATIONS
		weighted_final_observations = torch.mean(weight_obs*attention_values_obs, dim=-2).unsqueeze(-2).repeat(1,1,self.num_agents,1)

		# NODE FEATURES
		node_features = torch.cat([weighted_final_observations,weighted_attention_values], dim=-1)

		Value = F.leaky_relu(self.final_value_layer_1(node_features))
		Value = self.final_value_layer_2(Value)

		return Value, ret_weight_obs_actions, weight_obs



class MultiHeadAttentionCriticNetwork(nn.Module):
	def __init__(self, obs_act_input_dim, obs_act_output_dim, final_input_dim, final_output_dim, num_agents, num_actions, threshold=0.1, num_heads=2):
		super(MultiHeadAttentionCriticNetwork, self).__init__()
		
		self.num_agents = num_agents
		self.num_actions = num_actions
		# self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		self.device = "cpu"
		self.num_heads = num_heads

		self.key_networks = []
		self.query_networks = []
		self.attention_value_networks = []

		for i in range(self.num_heads):
			self.key_networks.append(nn.Linear(obs_act_input_dim, obs_act_output_dim, bias=False))
			self.query_networks.append(nn.Linear(obs_act_input_dim, obs_act_output_dim, bias=False))
			self.attention_value_networks.append(nn.Linear(obs_act_input_dim, obs_act_output_dim, bias=False))

		
		self.d_k = obs_act_output_dim


		# NOISE
		self.noise_normal = torch.distributions.Normal(loc=torch.tensor([0.0]), scale=torch.tensor([1.0]))
		self.noise_uniform = torch.rand
		# ********************************************************************************************************

		# ********************************************************************************************************
		# FCN FINAL LAYER TO GET VALUES
		self.final_value_layer_1 = nn.Linear(final_input_dim * self.num_heads, 64, bias=False)
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

		for i in range(self.num_heads):
			nn.init.xavier_uniform_(self.key_networks[i].weight)
			nn.init.xavier_uniform_(self.query_networks[i].weight)
			nn.init.xavier_uniform_(self.attention_value_networks[i].weight)


		nn.init.xavier_uniform_(self.final_value_layer_1.weight, gain=gain)
		nn.init.xavier_uniform_(self.final_value_layer_2.weight, gain=gain)



	def forward(self, states, policies, actions):

		# ATTENTION NETWORK FOR OBSERVATIONS AND ACTIONS
		# input to KEY, QUERY and ATTENTION VALUE NETWORK
		obs_actions = torch.cat([states,actions],dim=-1)
		# For calculating the right advantages
		obs_policy = torch.cat([states,policies], dim=-1)

		weighted_attention_values_list = []
		weight_obs_actions_list = []
		for i in range(self.num_heads):
			# KEYS
			key_obs_actions = self.key_networks[i](obs_actions)
			# QUERIES
			query_obs_actions = self.query_networks[i](obs_actions)
			# SCORE
			score_obs_actions = torch.bmm(query_obs_actions,key_obs_actions.transpose(1,2)).transpose(1,2).reshape(-1,1)
			score_obs_actions = score_obs_actions.reshape(-1,self.num_agents,1)
			# WEIGHT
			weight_obs_actions = F.softmax(score_obs_actions/math.sqrt(self.d_k), dim=-2)
			weight_obs_actions = weight_obs_actions.reshape(weight_obs_actions.shape[0]//self.num_agents,self.num_agents,-1)
			weight_obs_actions_list.append(weight_obs_actions)
			# MERGING OBSERVATION AND ACTIONS
			obs_actions_ = obs_actions.repeat(1,self.num_agents,1).reshape(obs_actions.shape[0],self.num_agents,self.num_agents,-1)
			obs_policy_ = obs_policy.repeat(1,self.num_agents,1).reshape(obs_policy.shape[0],self.num_agents,self.num_agents,-1)
			obs_actions_policies = self.place_policies*obs_policy_ + self.place_actions*obs_actions_
			# ATTENTION VALUES
			attention_values = torch.tanh(self.attention_value_networks[i](obs_actions_policies))
			attention_values_ = attention_values.repeat(1,self.num_agents,1,1).reshape(attention_values.shape[0],self.num_agents,self.num_agents,self.num_agents,-1)
			# WEIGHTED AVERAGE OF ATTENTION VALUES
			weight_obs_actions_ = weight_obs_actions.unsqueeze(-2).repeat(1,1,self.num_agents,1).unsqueeze(-1)
			uniform_noise = (self.noise_uniform((attention_values_.view(-1).size())).reshape(attention_values_.size()) - 0.5) * 0.1 #SCALING NOISE AND MAKING IT ZERO CENTRIC
			weighted_attention_values_list.append(torch.mean(attention_values_*weight_obs_actions_ + uniform_noise, dim=-2))


		weighted_attention_values_list = torch.cat([agg_values for agg_values in weighted_attention_values_list], dim=-1)

		Value = F.leaky_relu(self.final_value_layer_1(weighted_attention_values_list))
		Value = self.final_value_layer_2(Value)

		return Value, weight_obs_actions_list



class PolicyNetwork(nn.Module):
	def __init__(
		self,
		policy_sizes
		):
		super(PolicyNetwork,self).__init__()

		self.policy = create_model(policy_sizes)

	def forward(self,states):
		return F.softmax(self.policy(states),-1)




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
		node_features = torch.mean(weighted_attention_values, dim=-2)

		Policy = F.leaky_relu(self.final_policy_layer_1(node_features))
		Policy = F.softmax(self.final_policy_layer_2(Policy), dim=-1)

		return Policy, weight