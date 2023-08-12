from typing import Any, List, Tuple, Union
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import datetime
import math


class MLP_Policy(nn.Module):
	def __init__(self, obs_input_dim, num_actions, num_agents, device):
		super(MLP_Policy, self).__init__()

		self.name = "MLP Policy"

		self.rnn_hidden_state = None
		self.num_agents = num_agents
		self.num_actions = num_actions
		self.device = device
		self.Layer_1 = nn.Sequential(nn.Linear(obs_input_dim, 128), nn.GELU())
		self.RNN = nn.GRUCell(input_size=128, hidden_size=128)
		self.Layer_2 = nn.Sequential(nn.Linear(128, 64), nn.GELU(), nn.Linear(64, num_actions))

		self.reset_parameters()

	def reset_parameters(self):

		nn.init.xavier_uniform_(self.Layer_1[0].weight)
		nn.init.xavier_uniform_(self.Layer_2[0].weight)
		nn.init.xavier_uniform_(self.Layer_2[2].weight)


	def forward(self, local_observations, mask_actions=None):
		intermediate = self.Layer_1(local_observations)
		self.rnn_hidden_state = self.RNN(intermediate.view(-1, intermediate.shape[-1])).view(*intermediate.shape)
		policy = self.Layer_2(self.rnn_hidden_state) + mask_actions
		return F.softmax(policy, dim=-1), self.rnn_hidden_state


class TransformerCritic(nn.Module):
	'''
	https://proceedings.neurips.cc/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf
	'''
	def __init__(self, obs_input_dim, obs_act_input_dim, num_agents, num_actions, device):
		super(TransformerCritic, self).__init__()
		
		self.name = "TransformerCritic"

		self.num_agents = num_agents
		self.num_actions = num_actions
		self.device = device

		self.state_embed = nn.Sequential(
			nn.Linear(obs_input_dim, 128, bias=True), 
			nn.GELU(),
			)
		self.key = nn.Sequential(
			nn.Linear(128, 128, bias=True),
			nn.GELU(),
			)
		self.query = nn.Sequential(
			nn.Linear(128, 128, bias=True),
			nn.GELU(),
			)
		self.state_act_pol_embed = nn.Sequential(
			nn.Linear(obs_act_input_dim, 128, bias=True), 
			nn.GELU(),
			)
		self.attention_value = nn.Sequential(
			nn.Linear(128, 128, bias=True), 
			nn.GELU(),
			)

		# dimesion of key
		self.d_k_obs_act = 128 

		self.layer_norm_state_act_pol_embed = nn.LayerNorm(128) 

		self.curr_agent_state_embed = nn.Sequential(
			nn.Linear(obs_input_dim, 128, bias=True), 
			nn.GELU(),
			)
		self.layer_norm_state_embed = nn.LayerNorm(128)

		# ********************************************************************************************************

		# ********************************************************************************************************
		# FCN FINAL LAYER TO GET VALUES
		self.common_layer = nn.Sequential(
			nn.Linear(128*2, 128, bias=True), 
			nn.GELU()
			)
		self.rnn_hidden_state = None
		self.RNN = nn.GRUCell(128, 128)
		self.final_value_layers = nn.Sequential(
			nn.Linear(128, 128, bias=True), 
			nn.GELU(),
			nn.Linear(128, 1)
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

		# EMBEDDINGS
		nn.init.xavier_uniform_(self.state_embed[0].weight)
		nn.init.xavier_uniform_(self.state_act_pol_embed[0].weight)

		nn.init.xavier_uniform_(self.key[0].weight)
		nn.init.xavier_uniform_(self.query[0].weight)
		nn.init.xavier_uniform_(self.attention_value[0].weight)

		nn.init.xavier_uniform_(self.curr_agent_state_embed[0].weight)

		nn.init.xavier_uniform_(self.common_layer[0].weight)
		nn.init.xavier_uniform_(self.final_value_layers[0].weight)
		nn.init.xavier_uniform_(self.final_value_layers[2].weight)



	def forward(self, states, policies, actions):
		# EMBED STATES
		states_embed = self.state_embed(states)
		# KEYS
		key_obs = self.key(states_embed)
		# QUERIES
		query_obs = self.query(states_embed)
		# WEIGHT
		weight = F.softmax(torch.matmul(query_obs,key_obs.transpose(1,2))/math.sqrt(self.d_k_obs_act),dim=-1)
		ret_weight = weight

		obs_actions = torch.cat([states,actions],dim=-1)
		obs_policy = torch.cat([states,policies], dim=-1)
		obs_actions = obs_actions.repeat(1,self.num_agents,1).reshape(obs_actions.shape[0],self.num_agents,self.num_agents,-1)
		obs_policy = obs_policy.repeat(1,self.num_agents,1).reshape(obs_policy.shape[0],self.num_agents,self.num_agents,-1)
		obs_actions_policies = self.place_policies*obs_policy + self.place_actions*obs_actions
		# EMBED STATE ACTION POLICY
		obs_actions_policies_embed = self.state_act_pol_embed(obs_actions_policies)
		attention_values = self.attention_value(obs_actions_policies_embed)

		attention_values = attention_values.repeat(1,self.num_agents,1,1).reshape(attention_values.shape[0],self.num_agents,self.num_agents,self.num_agents,-1)

		weight = weight.unsqueeze(-2).repeat(1,1,self.num_agents,1).unsqueeze(-1)
		weighted_attention_values = attention_values*weight
		node_features = torch.sum(weighted_attention_values, dim=-2)
		node_features = self.layer_norm_state_act_pol_embed(obs_actions_policies_embed+node_features)

		curr_agent_state_embed = self.curr_agent_state_embed(states)
		curr_agent_state_embed = self.layer_norm_state_embed(curr_agent_state_embed+states_embed)
		curr_agent_node_features = torch.cat([curr_agent_state_embed.unsqueeze(-2).repeat(1,1,self.num_agents,1), node_features], dim=-1)

		intermediate = self.common_layer(curr_agent_node_features)
		self.rnn_hidden_state = self.RNN(intermediate.view(-1, intermediate.shape[-1])).reshape(*intermediate.shape)
		Value = self.final_value_layers(self.rnn_hidden_state)		

		return Value, ret_weight, self.rnn_hidden_state
