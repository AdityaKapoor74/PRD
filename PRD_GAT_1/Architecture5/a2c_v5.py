from typing import Any, List, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl 
import numpy as np
import dgl
import dgl.function as fn
from dgl import DGLGraph
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




class CriticNetwork(nn.Module):
	def __init__(self, obs_act_input_dim, obs_act_output_dim, final_input_dim, final_output_dim, num_agents, num_actions):
		super(CriticNetwork, self).__init__()
		
		self.num_agents = num_agents
		self.num_actions = num_actions
		# self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		self.device = "cpu"

		# *******************************************************************************************************
		# PROCESSING OF OBSERVATIONS
		# self.key_layer_1 = nn.Linear(obs_act_input_dim, 32, bias=False)
		# self.key_layer_2 = nn.Linear(32, obs_act_output_dim, bias=False)
		self.key_layer = nn.Linear(obs_act_input_dim, obs_act_output_dim, bias=False)

		# self.query_layer_1 = nn.Linear(obs_act_input_dim, 32, bias=False)
		# self.query_layer_2 = nn.Linear(32, obs_act_output_dim, bias=False)
		self.query_layer = nn.Linear(obs_act_input_dim, obs_act_output_dim, bias=False)

		self.attention_value_layer_1 = nn.Linear(obs_act_input_dim, 32, bias=False)
		# self.attention_value_layer_norm = nn.LayerNorm(32)
		self.attention_value_layer_2 = nn.Linear(32, obs_act_output_dim, bias=False)
		# self.attention_value_layer = nn.Linear(obs_act_input_dim, obs_act_output_dim, bias=False)

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

		# ********************************************************************************************************
		# Placing Policies instead of zs
		# self.place_policies = torch.zeros(self.num_agents,self.num_agents,self.num_agents,obs_act_input_dim).to(self.device)
		# self.place_actions = torch.ones(self.num_agents,self.num_agents,self.num_agents,obs_act_input_dim).to(self.device)
		# one_hots = torch.ones(obs_act_input_dim)
		# zero_hots = torch.zeros(obs_act_input_dim)

		# for i in range(self.num_agents):
		# 	for j in range(self.num_agents):
		# 		self.place_policies[i][j][j] = one_hots
		# 		self.place_actions[i][j][j] = zero_hots


		self.place_policies = torch.zeros(self.num_agents,self.num_agents,obs_act_input_dim).to(self.device)
		self.place_actions = torch.ones(self.num_agents,self.num_agents,obs_act_input_dim).to(self.device)
		one_hots = torch.ones(obs_act_input_dim)
		zero_hots = torch.zeros(obs_act_input_dim)

		for j in range(self.num_agents):
			self.place_policies[j][j] = one_hots
			self.place_actions[j][j] = zero_hots
		# ********************************************************************************************************* 

		self.reset_parameters()

	def reset_parameters(self):
		"""Reinitialize learnable parameters."""
		gain = nn.init.calculate_gain('leaky_relu')

		# nn.init.xavier_uniform_(self.key_layer_1.weight, gain=gain)
		# nn.init.xavier_uniform_(self.key_layer_2.weight, gain=gain)
		# nn.init.xavier_uniform_(self.query_layer_1.weight, gain=gain)
		# nn.init.xavier_uniform_(self.query_layer_2.weight, gain=gain)
		nn.init.xavier_uniform_(self.attention_value_layer_1.weight, gain=gain)
		nn.init.xavier_uniform_(self.attention_value_layer_2.weight, gain=gain)

		nn.init.xavier_uniform_(self.key_layer.weight, gain=gain)
		nn.init.xavier_uniform_(self.query_layer.weight, gain=gain)
		# nn.init.xavier_uniform_(self.attention_value_layer.weight, gain=gain)


		nn.init.xavier_uniform_(self.final_value_layer_1.weight, gain=gain)
		nn.init.xavier_uniform_(self.final_value_layer_2.weight, gain=gain)



	def forward(self, states, policies, actions):
		# print("STATES")
		# print(states)
		# print("ACTIONS")
		# print(actions)
		# print("POLICIES")
		# print(policies)

		# input to KEY, QUERY and ATTENTION VALUE NETWORK
		obs_actions = torch.cat([states,actions],dim=-1)
		# print("OBSERVATIONS ACTIONS")
		# print(obs_actions)

		# For calculating the right advantages
		obs_policy = torch.cat([states,policies], dim=-1)
		# print("OBSERVATIONS POLICIES")
		# print(obs_policy)

		# KEYS
		# key_obs_actions = F.leaky_relu(self.key_layer_1(obs_actions))
		# key_obs_actions = self.key_layer_2(key_obs_actions)
		key_obs_actions = self.key_layer(obs_actions)
		# print("KEYS")
		# print(key_obs_actions)
		# print("TRANSPOSE")
		# print(key_obs_actions.transpose(1,2))
		# QUERIES
		# query_obs_actions = F.leaky_relu(self.query_layer_1(obs_actions))
		# query_obs_actions = self.query_layer_2(query_obs_actions)
		query_obs_actions = self.query_layer(obs_actions)
		# print("QUERIES")
		# print(query_obs_actions)

		# print("SCORE")
		# print(torch.bmm(query_obs_actions,key_obs_actions.transpose(1,2)))

		score_obs_actions = torch.bmm(query_obs_actions,key_obs_actions.transpose(1,2)).transpose(1,2).reshape(-1,1)
		# print("SCORE")
		# print(score_obs_actions)

		score_obs_actions = score_obs_actions.reshape(-1,self.num_agents,1)
		# print("SCORE RESHAPE")
		# print(score_obs_actions)

		weight = F.softmax(score_obs_actions/math.sqrt(self.d_k_obs_act), dim=-2)
		weight = weight.reshape(weight.shape[0]//self.num_agents,self.num_agents,-1)
		ret_weight = weight
		weight = weight.unsqueeze(-2).repeat(1,1,self.num_agents,1).unsqueeze(-1)
		# print("WEIGHTS")
		# print(weight)
		
		obs_actions = obs_actions.repeat(1,self.num_agents,1).reshape(obs_actions.shape[0],self.num_agents,self.num_agents,-1)
		# obs_actions = obs_actions.repeat(1,1,self.num_agents,1).reshape(obs_actions.shape[0],self.num_agents,self.num_agents,self.num_agents,-1)
		# print("OBSERVATION ACTIONS INFLATED")
		# print(obs_actions)
		obs_policy = obs_policy.repeat(1,self.num_agents,1).reshape(obs_policy.shape[0],self.num_agents,self.num_agents,-1)
		# obs_policy = obs_policy.repeat(1,1,self.num_agents,1).reshape(obs_policy.shape[0],self.num_agents,self.num_agents,self.num_agents,-1)
		# print("OBSERVATION POLCIY INFLATED")
		# print(obs_policy)
		
		# print(self.place_policies)
		# print(self.place_actions)
		obs_actions_policies = self.place_policies*obs_policy + self.place_actions*obs_actions
		# print("OBSERVATION ACTIONS POLICIES")
		# print(obs_actions_policies)

		# attention_values = F.leaky_relu(self.attention_value_layer_1(obs_actions_policies))
		# attention_values = self.attention_value_layer_2(self.attention_value_layer_norm(attention_values))
		attention_values = torch.tanh(self.attention_value_layer_1(obs_actions_policies))
		attention_values = torch.tanh(self.attention_value_layer_2(attention_values))

		# normal_noise = self.noise_normal.sample((actions.view(-1).size())).reshape(actions.size())
		uniform_noise = self.noise_uniform((attention_values.view(-1).size())).reshape(attention_values.size())
		attention_values_noise = attention_values + uniform_noise

		attention_values_noise = attention_values_noise.repeat(1,self.num_agents,1,1).reshape(attention_values_noise.shape[0],self.num_agents,self.num_agents,self.num_agents,-1)
		# print("ATTENTION VALUES")
		# print(attention_values_noise)
		current_node_states = states.unsqueeze(-2).repeat(1,1,self.num_agents,1)
		# print("STATES INFLATED")
		# print(current_node_states)

		# print("AGGREGATING ATTENTION VALUES")
		# print(torch.mean(attention_values_noise*weight, dim=-2))
		node_features = torch.cat([current_node_states,torch.mean(attention_values_noise*weight, dim=-2)], dim=-1)

		Value = F.leaky_relu(self.final_value_layer_1(node_features))
		Value = self.final_value_layer_2(Value)

		return Value, ret_weight



class PolicyNetwork(nn.Module):
	def __init__(
		self,
		policy_sizes
		):
		super(PolicyNetwork,self).__init__()

		self.policy = create_model(policy_sizes)

	def forward(self,states):
		return F.softmax(self.policy(states),-1)