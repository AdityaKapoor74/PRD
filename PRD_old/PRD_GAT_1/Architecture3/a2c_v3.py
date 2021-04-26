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
	def __init__(self, obs_act_input_dim, obs_act_output_dim, encoder_input_dim, encoder_output_dim, final_input_dim, final_output_dim, num_agents, num_actions):
		super(CriticNetwork, self).__init__()
		
		self.num_agents = num_agents
		self.num_actions = num_actions
		# self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		self.device = "cpu"

		# *******************************************************************************************************
		# self.key_layer_1 = nn.Linear(obs_act_input_dim, 64, bias=True)
		# self.key_layer_2 = nn.Linear(64, obs_act_output_dim, bias=True)
		self.key_layer = nn.Linear(obs_act_input_dim, obs_act_output_dim, bias=False)

		# self.query_layer_1 = nn.Linear(obs_act_input_dim, 64, bias=True)
		# self.query_layer_2 = nn.Linear(64, obs_act_output_dim, bias=True)
		self.query_layer = nn.Linear(obs_act_input_dim, obs_act_output_dim, bias=False)

		# dimesion of key
		self.d_k = obs_act_output_dim

		# self.attention_value_layer_1 = nn.Linear(obs_act_input_dim, 64, bias=True)
		# self.attention_value_layer_2 = nn.Linear(64, obs_act_output_dim, bias=True)
		self.attention_value_layer = nn.Linear(obs_act_input_dim, obs_act_output_dim, bias=False)

		self.source_node_encoder = nn.Linear(encoder_input_dim, encoder_output_dim, bias=False)
		# self.source_node_encoders = []
		# self.encoder_output_dim = encoder_output_dim
		# for i in range(self.num_agents):
		# 	self.source_node_encoders.append(nn.Linear(encoder_input_dim, encoder_output_dim, bias=False))

		# self.final_value_layers = []
		# self.final_output_dim = final_output_dim
		# for i in range(self.num_agents):
		# 	self.final_value_layers.append(nn.Linear(final_input_dim, final_output_dim, bias=False))

		# ********************************************************************************************************

		# ********************************************************************************************************
		# FCN FINAL LAYER TO GET VALUES
		# self.final_value_layer_1 = nn.Linear(final_input_dim, 64, bias=True)
		# self.final_value_layer_2 = nn.Linear(64, final_output_dim, bias=True)
		self.final_value_layer = nn.Linear(final_input_dim, final_output_dim, bias=False)
		# ********************************************************************************************************

		# ********************************************************************************************************
		# Extracting source nodes observation,z values
		self.source_obsz = torch.zeros(self.num_agents,self.num_agents,obs_act_input_dim).to(self.device)
		one_hots = torch.ones(obs_act_input_dim)
		for j in range(self.num_agents):
			self.source_obsz[j][j] = one_hots
		# ********************************************************************************************************	

		# ********************************************************************************************************
		# Placing Policies instead of zs
		self.place_policies = torch.zeros(self.num_agents,self.num_agents,self.num_agents,obs_act_input_dim).to(self.device)
		self.place_actions = torch.ones(self.num_agents,self.num_agents,self.num_agents,obs_act_input_dim).to(self.device)
		one_hots = torch.ones(obs_act_input_dim)
		zero_hots = torch.zeros(obs_act_input_dim)

		for i in range(self.num_agents):
			for j in range(self.num_agents):
				self.place_policies[i][j][j] = one_hots
				self.place_actions[i][j][j] = zero_hots

		# ********************************************************************************************************* 


		# ********************************************************************************************************* 
		# PAIRINGS
		self.pairings = torch.zeros(self.num_agents,self.num_agents).to(self.device)
		for i in range(self.num_agents):
			for j in range(self.num_agents):
				# if i == self.num_agents-j-1:
				if i==j:
					self.pairings[i][j] = 1

		self.pairings = self.pairings.repeat(1,self.num_agents).reshape(self.num_agents,self.num_agents,self.num_agents).unsqueeze(-1)

		# self.reset_parameters()

	def reset_parameters(self):
		"""Reinitialize learnable parameters."""
		gain = nn.init.calculate_gain('leaky_relu')

		nn.init.xavier_uniform_(self.key_layer_1.weight, gain=gain)
		nn.init.xavier_uniform_(self.key_layer_2.weight, gain=gain)
		nn.init.xavier_uniform_(self.query_layer_1.weight, gain=gain)
		nn.init.xavier_uniform_(self.query_layer_2.weight, gain=gain)
		nn.init.xavier_uniform_(self.attention_value_layer_1.weight, gain=gain)
		nn.init.xavier_uniform_(self.attention_value_layer_2.weight, gain=gain)
		nn.init.xavier_uniform_(self.source_node_encoder.weight, gain=gain)

		nn.init.xavier_uniform_(self.final_value_layer_1.weight, gain=gain)
		nn.init.xavier_uniform_(self.final_value_layer_2.weight, gain=gain)


	def forward(self, observations, policies, actions):
		# equation (1)
		obs_actions = torch.cat([observations,actions],dim=-1)

		# key = F.leaky_relu(self.key_layer_1(obs_actions))
		# key = self.key_layer_2(key)
		key = self.key_layer(obs_actions)

		# query = F.leaky_relu(self.query_layer_1(obs_actions))
		# query = self.query_layer_2(query)
		query = self.query_layer(obs_actions)

		# We want row i to have scores of every edge corresponding to node i
		scores = torch.bmm(query,key.transpose(1,2)).transpose(1,2).reshape(-1,1)

		# for softmax
		scores = scores.reshape(-1,self.num_agents,1)
		# weights = torch.softmax(torch.exp((scores / math.sqrt(self.d_k)).clamp(-5, 5)), dim=-2)

		weights = torch.sigmoid(scores/math.sqrt(self.d_k))

		obs_actions = obs_actions.repeat(1,self.num_agents,1).reshape(obs_actions.shape[0],self.num_agents,self.num_agents,-1)
		obs_actions = obs_actions.repeat(1,self.num_agents,1,1).reshape(obs_actions.shape[0],self.num_agents,self.num_agents,self.num_agents,-1)

		obs = observations
		obs_policy = torch.cat([obs,policies], dim=-1)
		obs_policy = obs_policy.repeat(1,self.num_agents,1).reshape(obs_policy.shape[0],self.num_agents,self.num_agents,-1)
		obs_policy = obs_policy.repeat(1,self.num_agents,1,1).reshape(obs_policy.shape[0],self.num_agents,self.num_agents,self.num_agents,-1)

		obs_actions_policies = obs_policy*self.place_policies + obs_actions*self.place_actions
		# print(obs_actions_policies)

		# attention_value = F.leaky_relu(self.attention_value_layer_1(obs_actions_policies))
		# attention_value = self.attention_value_layer_2(attention_value)
		attention_value = self.attention_value_layer(obs_actions_policies)

		weights = weights.reshape(weights.shape[0]//self.num_agents,self.num_agents,-1).unsqueeze(-1)
		weights = weights.repeat(1,1,self.num_agents,1).reshape(weights.shape[0],self.num_agents,self.num_agents,self.num_agents,1)

		aggregation = attention_value*weights
		# aggregation = attention_value*self.pairings
		aggregation = torch.mean(aggregation,dim=-2)

		# observation = self.source_node_encoder(obs)
		# observation = observation.reshape(observation.shape[0]//self.num_agents,self.num_agents,-1).unsqueeze(-2)
		# observation = observation.repeat(1,1,self.num_agents,1)
		# obs_encoded = torch.zeros(obs.shape[0],obs.shape[1],self.encoder_output_dim)
		# for i in range(self.num_agents):
		# 	obs_encoded[:,i,:] = self.source_node_encoders[i](obs[:,i,:])


		obs_encoded = self.source_node_encoder(obs)

		obs = obs.unsqueeze(-2).repeat(1,1,self.num_agents,1)
		obs_encoded = obs_encoded.unsqueeze(-2).repeat(1,1,self.num_agents,1)
		
		# node_features = torch.cat([observation,aggregation],dim=-1)
		node_features = torch.cat([obs_encoded,aggregation],dim=-1)
		# print(node_features.shape)

		# Value = F.leaky_relu(self.final_value_layer_1(node_features))
		# Value = self.final_value_layer_2(Value)
		# Value = torch.zeros(node_features.shape[0],node_features.shape[1],node_features.shape[2],self.final_output_dim)

		# for i in range(self.num_agents):
		# 	Value[:,i,:,:] = self.final_value_layers[i](node_features[:,i,:,:])
		Value = self.final_value_layer(node_features)


		return Value, weights



class PolicyNetwork(nn.Module):
	def __init__(
		self,
		policy_sizes
		):
		super(PolicyNetwork,self).__init__()

		self.policy = create_model(policy_sizes)

	def forward(self,states):
		return F.softmax(self.policy(states),-1)