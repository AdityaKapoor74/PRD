from typing import Any, List, Tuple, Union
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


'''
Scalar Dot Product Attention: Transformer
'''

class TransformerCritic(nn.Module):
	'''
	https://proceedings.neurips.cc/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf
	'''
	def __init__(self, obs_input_dim, obs_output_dim, obs_act_input_dim, obs_act_output_dim, final_input_dim, final_output_dim, num_agents, num_actions, device):
		super(TransformerCritic, self).__init__()
		
		self.name = "TransformerCritic"

		self.num_agents = num_agents
		self.num_actions = num_actions
		self.device = device

		self.state_act_embed_attn = nn.Sequential(nn.Linear(obs_act_input_dim, 128), nn.LeakyReLU())
		self.state_embed_attn = nn.Sequential(nn.Linear(obs_input_dim, 128), nn.LeakyReLU())
		self.key_layer = nn.Linear(128, obs_act_output_dim, bias=True)
		self.query_layer = nn.Linear(128, obs_output_dim, bias=True)
		self.attention_value_layer = nn.Linear(128, obs_act_output_dim, bias=True)
		# dimesion of key
		self.d_k_obs_act = obs_act_output_dim  

		# score corresponding to current agent should be 0
		self.mask_score = torch.ones(self.num_agents,self.num_agents, dtype=torch.bool).to(self.device)
		self.mask_attn_values = torch.ones(self.num_agents,self.num_agents, 128, dtype=torch.bool).to(self.device)
		mask = torch.zeros(128, dtype=torch.bool).to(self.device)
		for j in range(self.num_agents):
			self.mask_score[j][j] = False
			self.mask_attn_values[j][j] = mask

		# ********************************************************************************************************

		# ********************************************************************************************************
		# EMBED S of agent whose Q value is being est
		self.state_embed_q = nn.Sequential(nn.Linear(obs_input_dim, obs_output_dim), nn.LeakyReLU())
		# FCN FINAL LAYER TO GET VALUES
		self.rnn_hidden_state = None
		self.RNN = nn.GRUCell(final_input_dim, final_input_dim)
		self.final_value_layers = nn.Sequential(
												nn.Linear(final_input_dim, 64, bias=True),
												nn.LeakyReLU(),
												nn.Linear(64, final_output_dim, bias=True)
												)
		# ********************************************************************************************************
		self.reset_parameters()


	def reset_parameters(self):
		"""Reinitialize learnable parameters."""
		gain_leaky = nn.init.calculate_gain('leaky_relu')

		nn.init.xavier_uniform_(self.state_act_embed_attn[0].weight, gain=gain_leaky)
		nn.init.xavier_uniform_(self.state_embed_attn[0].weight, gain=gain_leaky)
		nn.init.xavier_uniform_(self.state_embed_q[0].weight, gain=gain_leaky)

		nn.init.xavier_uniform_(self.key_layer.weight)
		nn.init.xavier_uniform_(self.query_layer.weight)
		nn.init.xavier_uniform_(self.attention_value_layer.weight)

		nn.init.xavier_uniform_(self.final_value_layers[0].weight, gain=gain_leaky)
		nn.init.xavier_uniform_(self.final_value_layers[2].weight, gain=gain_leaky)


	def forward(self, states, actions):
		state_actions = torch.cat([states, actions], dim=-1)
		state_embed_attn = self.state_embed_attn(states)
		state_act_embed_attn = self.state_act_embed_attn(state_actions)
		# Keys
		keys = self.key_layer(state_act_embed_attn)
		# Queries
		queries = self.query_layer(state_embed_attn)
		# Calc score (score corresponding to self to be made 0)
		score = torch.matmul(queries, keys.transpose(1,2))/math.sqrt(self.d_k_obs_act)
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
		shape = node_features.shape
		self.rnn_hidden_state = self.RNN(node_features.view(-1, shape[-1])).reshape(*shape)
		Value = self.final_value_layers(self.rnn_hidden_state)

		return Value, weight, self.rnn_hidden_state
