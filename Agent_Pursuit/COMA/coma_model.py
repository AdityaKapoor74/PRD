from typing import Any, List, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import datetime
import math


class MLPPolicy(nn.Module):
	def __init__(self, height, width, in_channel, num_agents, num_actions):
		super(MLPPolicy, self).__init__()

		self.name = "MLP Policy"

		self.num_agents = num_agents
		self.num_actions = num_actions
		self.obs_input_dim = height*width*in_channel

		scale = (height*width) ** -0.5
		self.positions = nn.Parameter(scale * torch.randn(self.num_agents, self.obs_input_dim)) # Num Patches, embedding size
		self.Conv = nn.Conv2d(in_channel, self.obs_input_dim, height)
		self.Policy_MLP = nn.Sequential(
			nn.Linear(self.obs_input_dim, 128),
			nn.GELU(),
			nn.Linear(128, 64),
			nn.GELU(),
			nn.Linear(64, num_actions),
			nn.Softmax(dim=-1)
			)

		self.reset_parameters()

	def reset_parameters(self):
		# gain = nn.init.calculate_gain('tanh', 0.1)
		# gain_last_layer = nn.init.calculate_gain('tanh', 0.01)

		nn.init.orthogonal_(self.Conv.weight)
		nn.init.xavier_uniform_(self.Policy_MLP[0].weight)
		nn.init.xavier_uniform_(self.Policy_MLP[2].weight)
		nn.init.xavier_uniform_(self.Policy_MLP[4].weight)


	def forward(self, local_observations):
		local_observations = self.Conv(local_observations).reshape(-1, self.num_agents, self.obs_input_dim)+self.positions
		return self.Policy_MLP(local_observations)


'''
Scalar Dot Product Attention: Transformer
'''

class TransformerCritic(nn.Module):
	'''
	https://proceedings.neurips.cc/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf
	'''
	def __init__(self, height, width, in_channel, num_agents, num_actions, device):
		super(TransformerCritic, self).__init__()
		
		self.name = "TransformerCritic"

		self.num_agents = num_agents
		self.num_actions = num_actions
		self.device = device

		self.Conv_flat = nn.Conv2d(in_channel, height*width*in_channel, height)

		obs_input_dim = height*width*in_channel
		self.obs_input_dim = obs_input_dim

		scale = (height*width) ** -0.5
		self.positions = nn.Parameter(scale * torch.randn(self.num_agents, self.obs_input_dim)) # Num Patches, embedding size

		self.state_act_embed_attn = nn.Sequential(nn.Linear(obs_input_dim+self.num_actions, 64), nn.LeakyReLU())
		self.state_embed_attn = nn.Sequential(nn.Linear(obs_input_dim, 64), nn.LeakyReLU())
		self.key_layer = nn.Linear(64, 64, bias=True)
		self.query_layer = nn.Linear(64, 64, bias=True)
		self.attention_value_layer = nn.Linear(64, 64, bias=True)
		# dimesion of key
		self.d_k_obs_act = 64  

		# score corresponding to current agent should be 0
		self.mask_score = torch.ones(self.num_agents,self.num_agents, dtype=torch.bool).to(self.device)
		self.mask_attn_values = torch.ones(self.num_agents,self.num_agents, 64, dtype=torch.bool).to(self.device)
		mask = torch.zeros(64, dtype=torch.bool).to(self.device)
		for j in range(self.num_agents):
			self.mask_score[j][j] = False
			self.mask_attn_values[j][j] = mask

		# ********************************************************************************************************

		# ********************************************************************************************************
		# EMBED S of agent whose Q value is being est
		self.state_embed_q = nn.Sequential(nn.Linear(obs_input_dim, 64), nn.LeakyReLU())
		# FCN FINAL LAYER TO GET VALUES
		self.final_value_layers = nn.Sequential(
												nn.Linear(64+64, 64, bias=True),
												nn.LeakyReLU(),
												nn.Linear(64, self.num_actions, bias=True)
												)
		# ********************************************************************************************************
		self.reset_parameters()


	def reset_parameters(self):
		"""Reinitialize learnable parameters."""
		gain_leaky = nn.init.calculate_gain('leaky_relu')

		nn.init.orthogonal_(self.Conv_flat.weight)

		nn.init.xavier_uniform_(self.state_act_embed_attn[0].weight, gain=gain_leaky)
		nn.init.xavier_uniform_(self.state_embed_attn[0].weight, gain=gain_leaky)
		nn.init.xavier_uniform_(self.state_embed_q[0].weight, gain=gain_leaky)

		nn.init.xavier_uniform_(self.key_layer.weight)
		nn.init.xavier_uniform_(self.query_layer.weight)
		nn.init.xavier_uniform_(self.attention_value_layer.weight)

		nn.init.xavier_uniform_(self.final_value_layers[0].weight, gain=gain_leaky)
		nn.init.xavier_uniform_(self.final_value_layers[2].weight, gain=gain_leaky)


	def forward(self, states, actions):
		states = self.Conv_flat(states).reshape(-1, self.num_agents, self.obs_input_dim) + self.positions

		state_actions = torch.cat([states, actions], dim=-1)
		state_embed_attn = self.state_embed_attn(states)
		state_act_embed_attn = self.state_act_embed_attn(state_actions)
		# Keys
		keys = self.key_layer(state_act_embed_attn)
		# Queries
		queries = self.query_layer(state_embed_attn)
		# Calc score (score corresponding to self to be made 0)
		score = torch.matmul(queries,keys.transpose(1,2))/math.sqrt(self.d_k_obs_act)
		mask = torch.ones_like(score, dtype=torch.bool).to(self.device)*self.mask_score
		score = score[mask].reshape(mask.shape[0], self.num_agents,-1)
		weight = F.softmax(score,dim=-1)
		attention_values = self.attention_value_layer(state_act_embed_attn).unsqueeze(1).repeat(1,self.num_agents,1,1)
		mask_attn = torch.ones_like(attention_values, dtype=torch.bool).to(self.device) * self.mask_attn_values
		attention_values = attention_values[mask_attn].reshape(mask.shape[0], self.num_agents, self.num_agents-1,-1)
		x = torch.sum(attention_values*weight.unsqueeze(-1), dim=-2)

		# Embedding state of current agent
		curr_agent_state_action_embed = self.state_embed_q(states)

		node_features = torch.cat([curr_agent_state_action_embed, x], dim=-1)

		Value = self.final_value_layers(node_features)

		return Value, weight