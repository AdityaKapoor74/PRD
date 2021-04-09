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
	def __init__(self, weight_input_dim, weight_output_dim, obs_z_input_dim, obs_z_output_dim, final_input_dim, final_output_dim, num_agents, num_actions):
		super(CriticNetwork, self).__init__()
		
		self.num_agents = num_agents
		self.num_actions = num_actions
		# self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		self.device = "cpu"
		self.z = None
		self.weight_z = None

		# *******************************************************************************************************
		# WEIGHT NETWORK
		self.key_weight_layer_1 = nn.Linear(weight_input_dim, 64, bias=True)
		self.key_weight_layer_2 = nn.Linear(64, weight_output_dim, bias=True)
		# self.key_fc_layer = nn.Linear(weight_input_dim, weight_output_dim, bias=True)

		self.query_weight_layer_1 = nn.Linear(weight_input_dim, 64, bias=True)
		self.query_weight_layer_2 = nn.Linear(64, weight_output_dim, bias=True)
		# self.query_fc_layer = nn.Linear(weight_input_dim, weight_output_dim, bias=True)

		# dimesion of key
		self.d_k_weight = weight_output_dim
		# ********************************************************************************************************

		# *******************************************************************************************************
		# PROCESSING OF OBSERVATIONS
		self.key_obsz_layer_1 = nn.Linear(obs_z_input_dim, 64, bias=True)
		self.key_obsz_layer_2 = nn.Linear(64, obs_z_output_dim, bias=True)
		# self.key_fc_layer = nn.Linear(obs_z_input_dim, obs_z_output_dim, bias=True)

		self.query_obsz_layer_1 = nn.Linear(obs_z_input_dim, 64, bias=True)
		self.query_obsz_layer_2 = nn.Linear(64, obs_z_output_dim, bias=True)
		# self.query_fc_layer = nn.Linear(obs_z_input_dim, obs_z_output_dim, bias=True)

		self.attention_value_obsz_layer_1 = nn.Linear(obs_z_input_dim, 64, bias=True)
		self.attention_value_obsz_layer_2 = nn.Linear(64, obs_z_output_dim, bias=True)
		# self.attention_value_obsz_layer = nn.Linear(obs_z_input_dim, obs_z_output_dim, bias=True)


		# dimesion of key
		self.d_k_obsz = obs_z_output_dim
		# ********************************************************************************************************

		# ********************************************************************************************************
		# FCN FINAL LAYER TO GET VALUES
		self.final_value_layer_1 = nn.Linear(final_input_dim, 64, bias=True)
		self.final_value_layer_2 = nn.Linear(64, final_output_dim, bias=True)
		# ********************************************************************************************************

		# ********************************************************************************************************
		# Extracting source nodes observation,z values
		self.source_obsz = torch.zeros(self.num_agents,self.num_agents,obs_z_input_dim).to(self.device)
		one_hots = torch.ones(obs_z_input_dim)
		for j in range(self.num_agents):
			self.source_obsz[j][j] = one_hots
		# ********************************************************************************************************	

		# ********************************************************************************************************
		# Placing Policies instead of zs
		self.place_policies = torch.zeros(self.num_agents,self.num_agents,self.num_agents,obs_z_input_dim).to(self.device)
		self.place_zs = torch.ones(self.num_agents,self.num_agents,self.num_agents,obs_z_input_dim).to(self.device)
		one_hots = torch.ones(obs_z_input_dim)
		zero_hots = torch.zeros(obs_z_input_dim)

		for i in range(self.num_agents):
			for j in range(self.num_agents):
				self.place_policies[i][j][j] = one_hots
				self.place_zs[i][j][j] = zero_hots

		# self.place_policies = self.place_policies.reshape(self.num_agents,-1,obs_z_input_dim)
		# self.place_zs = self.place_zs.reshape(self.num_agents,-1,obs_z_input_dim)

		# ********************************************************************************************************* 

		# self.reset_parameters()

	def reset_parameters(self):
		"""Reinitialize learnable parameters."""
		gain = nn.init.calculate_gain('leaky_relu')

		nn.init.xavier_uniform_(self.key_weight_layer_1.weight, gain=gain)
		nn.init.xavier_uniform_(self.key_weight_layer_2.weight, gain=gain)
		nn.init.xavier_uniform_(self.query_weight_layer_1.weight, gain=gain)
		nn.init.xavier_uniform_(self.query_weight_layer_2.weight, gain=gain)

		nn.init.xavier_uniform_(self.key_obsz_layer_1.weight, gain=gain)
		nn.init.xavier_uniform_(self.key_obsz_layer_2.weight, gain=gain)
		nn.init.xavier_uniform_(self.query_obsz_layer_1.weight, gain=gain)
		nn.init.xavier_uniform_(self.query_obsz_layer_2.weight, gain=gain)
		nn.init.xavier_uniform_(self.attention_value_obsz_layer_1.weight, gain=gain)
		nn.init.xavier_uniform_(self.attention_value_obsz_layer_2.weight, gain=gain)

		nn.init.xavier_uniform_(self.final_value_layer_1.weight, gain=gain)
		nn.init.xavier_uniform_(self.final_value_layer_2.weight, gain=gain)



	def forward(self, states, policies, actions):
		# equation (1)
		key_z = F.leaky_relu(self.key_weight_layer_1(states))
		key_z = self.key_weight_layer_2(key_z)

		query_z = F.leaky_relu(self.query_weight_layer_1(states))
		query_z = self.query_weight_layer_2(query_z)


		scores_z = torch.bmm(query_z,key_z.transpose(1,2)).transpose(1,2).reshape(-1,1)
		scores_z = scores_z.reshape(-1,self.num_agents,1)

		weight_z = torch.sigmoid(scores_z/math.sqrt(self.d_k_weight))
		weight_z = weight_z.reshape(weight_z.shape[0]//self.num_agents,self.num_agents,-1).unsqueeze(-1)

		policies = policies.repeat(1,self.num_agents,1).reshape(policies.shape[0],self.num_agents,self.num_agents,-1)
		actions = actions.repeat(1,self.num_agents,1).reshape(actions.shape[0],self.num_agents,self.num_agents,-1)
		z = weight_z*actions + (1-weight_z)*policies

		states = states.repeat(1,self.num_agents,1).reshape(states.shape[0],self.num_agents,self.num_agents,-1)

		obs_z = torch.cat([states,z],dim=-1)

		obs_z_shape = obs_z.shape
		source_obsz = self.source_obsz
		source_obsz = source_obsz.repeat(obs_z.shape[0],1,1).reshape(obs_z_shape)

		# storing obs_z for every node --> obs_1, z_11; obs_2, z_21 ...
		source_obs_z = torch.sum(source_obsz * obs_z, dim=-2) #to match dimensions
		source_obs_z = source_obs_z.repeat(1,1,self.num_agents).reshape(obs_z_shape)

		key_obsz = F.leaky_relu(self.key_obsz_layer_1(source_obs_z))
		key_obsz = self.key_obsz_layer_2(key_obsz)

		query_obsz = F.leaky_relu(self.query_obsz_layer_1(obs_z))
		query_obsz = self.query_obsz_layer_2(query_obsz)


		scores_obsz = torch.sum(key_obsz*query_obsz, dim=-1)

		# weight_obsz = torch.softmax(torch.exp((score_obsz / math.sqrt(self.d_k_obsz)).clamp(-5, 5)), dim=-1).unsqueeze(-1)
		weight_obsz = torch.softmax(scores_obsz/math.sqrt(self.d_k_obsz), dim=-1)
		ret_weight_obsz = weight_obsz.unsqueeze(-1)

		# inflating the dimensions to include policies so that we can calculate V(i,j) = Estimated return for agent i not conditioned on agent j's actions
		obs_z = obs_z.repeat(1,1,self.num_agents,1).reshape(obs_z.shape[0],self.num_agents,self.num_agents,self.num_agents,-1)
		# doing these operations to get timesteps x num_agents x num_agents x num_agents x dimension
		place_policies = torch.cat([states,policies],dim=-1)
		place_policies = place_policies.repeat(1,self.num_agents,1,1).reshape(place_policies.shape[0],self.num_agents,self.num_agents,self.num_agents,-1)
		obs_zs = obs_z*self.place_zs
		obs_pis = place_policies*self.place_policies
		obs_z_pi = obs_zs + obs_pis

		attention_value_other_obsz = F.leaky_relu(self.attention_value_obsz_layer_1(obs_z_pi))
		attention_value_other_obsz = self.attention_value_obsz_layer_2(attention_value_other_obsz)

		weight_obsz = weight_obsz.unsqueeze(-2).repeat(1,1,self.num_agents,1).reshape(weight_obsz.shape[0],self.num_agents,self.num_agents,self.num_agents,-1)
		
		node_features = torch.mean(attention_value_other_obsz * weight_obsz, dim=-2)

		node_features = torch.cat([states,node_features], dim=-1)

		Value = F.leaky_relu(self.final_value_layer_1(node_features))
		Value = self.final_value_layer_2(Value)


		return Value, weight_z, ret_weight_obsz



class PolicyNetwork(nn.Module):
	def __init__(
		self,
		policy_sizes
		):
		super(PolicyNetwork,self).__init__()

		self.policy = create_model(policy_sizes)

	def forward(self,states):
		return F.softmax(self.policy(states),-1)