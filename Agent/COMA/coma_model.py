from typing import Any, List, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import datetime
import math


def init(module, weight_init, bias_init, gain=1):
	weight_init(module.weight.data, gain=gain)
	if module.bias is not None:
		bias_init(module.bias.data)
	return module

def init_(m, gain=0.01, activate=False):
	if activate:
		gain = nn.init.calculate_gain('relu')
	return init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), gain=gain)


class RNN_Policy(nn.Module):
	def __init__(self, obs_input_dim, num_actions, rnn_num_layers, num_agents, device):
		super(RNN_Policy, self).__init__()

		self.name = "RNN Policy"

		self.num_agents = num_agents
		self.rnn_num_layers = rnn_num_layers
		self.num_actions = num_actions
		self.device = device
		self.mask_value = torch.tensor(
				torch.finfo(torch.float).min, dtype=torch.float
			)

		self.Layer_1 = nn.Sequential(
			init_(nn.Linear(obs_input_dim+num_actions, 64)), 
			nn.GELU(),
			nn.LayerNorm(64),
			)
		self.RNN = nn.GRU(input_size=64, hidden_size=64, num_layers=rnn_num_layers, batch_first=True)
		self.Layer_2 = nn.Sequential(
			nn.LayerNorm(64),
			init_(nn.Linear(64, num_actions)),
			)

		for name, param in self.RNN.named_parameters():
			if 'bias' in name:
				nn.init.constant_(param, 0)
			elif 'weight' in name:
				nn.init.orthogonal_(param)


	# def forward(self, local_observations, mask_actions=None):
	# 	intermediate = self.Layer_1(local_observations)
	# 	if self.rnn_hidden_state is not None:
	# 		self.rnn_hidden_state = self.RNN(intermediate.view(-1, intermediate.shape[-1]), self.rnn_hidden_state.view(-1, intermediate.shape[-1])).view(*intermediate.shape)
	# 	else:
	# 		self.rnn_hidden_state = self.RNN(intermediate.view(-1, intermediate.shape[-1]), self.rnn_hidden_state).view(*intermediate.shape)
	# 	policy = self.Layer_2(self.rnn_hidden_state) + mask_actions
	# 	return F.softmax(policy, dim=-1), self.rnn_hidden_state

	def forward(self, local_observations, hidden_state, mask_actions=None):
		batch, timesteps, num_agents, _ = local_observations.shape
		intermediate = self.Layer_1(local_observations)
		intermediate = intermediate.permute(0, 2, 1, 3).reshape(batch*num_agents, timesteps, -1)
		hidden_state = hidden_state.reshape(self.rnn_num_layers, batch*num_agents, -1)
		output, h = self.RNN(intermediate, hidden_state)
		output = output.reshape(batch, num_agents, timesteps, -1).permute(0, 2, 1, 3)
		logits = self.Layer_2(output)

		logits = torch.where(mask_actions, logits, self.mask_value)
		return F.softmax(logits, dim=-1), h


'''
Scalar Dot Product Attention: Transformer
'''

class TransformerCritic(nn.Module):
	'''
	https://proceedings.neurips.cc/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf
	'''
	def __init__(self, obs_input_dim, obs_output_dim, obs_act_input_dim, obs_act_output_dim, final_input_dim, final_output_dim, num_agents, rnn_num_layers, num_actions, device):
		super(TransformerCritic, self).__init__()
		
		self.name = "TransformerCritic"

		self.num_agents = num_agents
		self.num_actions = num_actions
		self.device = device

		self.state_act_embed_attn = nn.Sequential(
			init_(nn.Linear(obs_act_input_dim, 64), activate=True), 
			nn.GELU(),
			nn.LayerNorm(64),
			)
		self.state_embed_attn = nn.Sequential(
			init_(nn.Linear(obs_input_dim, 64), activate=True), 
			nn.GELU(),
			nn.LayerNorm(64),
			)

		self.key_layer = init_(nn.Linear(64, obs_act_output_dim, bias=False))
		self.query_layer = init_(nn.Linear(64, obs_output_dim, bias=False))
		self.attention_value_layer = init_(nn.Linear(64, obs_act_output_dim, bias=False))
		self.attention_value_dropout = nn.Dropout(0.2)
		self.attention_value_layer_norm = nn.LayerNorm(obs_act_output_dim)

		self.attention_value_linear = nn.Sequential(
			init_(nn.Linear(obs_act_output_dim, 64), activate=True),
			nn.Dropout(0.2),
			nn.GELU(),
			nn.LayerNorm(64),
			init_(nn.Linear(64, 64))
			)
		self.attention_value_linear_dropout = nn.Dropout(0.2)

		self.attention_value_linear_layer_norm = nn.LayerNorm(64)

		# dimesion of key
		self.d_k_obs_act = obs_act_output_dim  

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
		self.state_embed_q = nn.Sequential(
			init_(nn.Linear(obs_input_dim, obs_output_dim), activate=True), 
			nn.GELU(),
			nn.LayerNorm(obs_output_dim),
			)

		self.common_layer = nn.Sequential(
			init_(nn.Linear(obs_output_dim+64, 64), activate=True),
			nn.GELU()
			)

		self.RNN = nn.GRU(input_size=64, hidden_size=64, num_layers=rnn_num_layers, batch_first=True)
		self.final_value_layers = nn.Sequential(
								nn.LayerNorm(64),
								init_(nn.Linear(64, final_output_dim, bias=True), activate=True)
								)


	def forward(self, states, actions, rnn_hidden_state):
		batch, timesteps, num_agents, _ = states.shape

		states = states.reshape(batch*timesteps, num_agents, -1)
		actions = actions.reshape(batch*timesteps, num_agents, -1)
		state_actions = torch.cat([states, actions], dim=-1)
		state_embed_attn = self.state_embed_attn(states)
		state_act_embed_attn = self.state_act_embed_attn(state_actions)
		# Keys
		keys = self.key_layer(state_embed_attn)
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
		x = self.attention_value_layer_norm(self.attention_value_dropout(x))
		x_ = self.attention_value_linear(x)
		x_ = self.attention_value_linear_dropout(x_)
		x = self.attention_value_linear_layer_norm(x+x_)

		# Embedding state of current agent
		curr_agent_state_embed = self.state_embed_q(states)
		node_features = torch.cat([curr_agent_state_embed, x], dim=-1)
		node_features = self.common_layer(node_features).reshape(batch, timesteps, num_agents, -1).permute(0, 2, 1, 3).reshape(batch*num_agents, timesteps, -1)
		rnn_output, rnn_hidden_state = self.RNN(node_features, rnn_hidden_state)
		rnn_output = rnn_output.reshape(batch, num_agents, timesteps, -1).permute(0, 2, 1, 3).reshape(batch*timesteps, num_agents, -1)
		Value = self.final_value_layers(rnn_output)

		
		return Value, weight, rnn_hidden_state
