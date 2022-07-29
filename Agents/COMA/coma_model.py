from typing import Any, List, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import datetime
import math


class MLPPolicy(nn.Module):
	def __init__(self, in_channels, obs_input_dim, num_actions, num_agents, device):
		super(MLPPolicy,self).__init__()

		self.name = "MLPPolicy"
		self.num_agents = num_agents		
		self.device = device

		self.Policy_CNN = nn.Sequential(
			nn.Conv2d(in_channels, 16, kernel_size=3, stride=1, padding=0, bias=True),
			nn.LeakyReLU(),
			nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=0, bias=True),
			nn.LeakyReLU()
			)
		self.Policy_MLP = nn.Sequential(
			nn.Linear(32+obs_input_dim*self.num_agents, 128),
			nn.Tanh(),
			nn.Linear(128, 64),
			nn.Tanh(),
			nn.Linear(64, num_actions),
			nn.Softmax(dim=-1)
			)

		self.reset_parameters()

	def reset_parameters(self):
		gain = nn.init.calculate_gain('tanh')
		gain_last_layer = nn.init.calculate_gain('tanh', 0.01)

		nn.init.orthogonal_(self.Policy_CNN[0].weight, gain=gain)
		nn.init.orthogonal_(self.Policy_CNN[2].weight, gain=gain)

		nn.init.orthogonal_(self.Policy_MLP[0].weight, gain=gain)
		nn.init.orthogonal_(self.Policy_MLP[2].weight, gain=gain)
		nn.init.orthogonal_(self.Policy_MLP[4].weight, gain=gain_last_layer)


	def forward(self, local_observations, agent_global_positions, agent_one_hot_encoding):
		agent_locs_ids = torch.cat([agent_global_positions, agent_one_hot_encoding], dim=-1)
		agent_states_shape = local_observations.shape
		intermediate_output = self.Policy_CNN(local_observations.reshape(-1,agent_states_shape[-3],agent_states_shape[-2],agent_states_shape[-1])).reshape(local_observations.shape[0], -1)
		if agent_states_shape[0] == self.num_agents:
			states_aug = torch.stack([torch.roll(agent_locs_ids,-i,0) for i in range(self.num_agents)], dim=0).transpose(1,0)
			intermediate_output = torch.cat([intermediate_output.reshape(self.num_agents,-1), states_aug.reshape(self.num_agents, -1)], dim=-1).to(self.device)
		else:
			states_aug = torch.stack([torch.roll(agent_locs_ids,-i,1) for i in range(self.num_agents)], dim=0).transpose(1,0)
			intermediate_output = torch.cat([intermediate_output.reshape(agent_states_shape[0],agent_states_shape[1],-1), states_aug.reshape(agent_locs_ids.shape[0],self.num_agents,-1)], dim=-1).to(self.device)
		return self.Policy_MLP(intermediate_output)


'''
Scalar Dot Product Attention: Transformer
'''

class TransformerCritic(nn.Module):
	'''
	https://proceedings.neurips.cc/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf
	'''
	def __init__(self, in_channels, obs_input_dim, obs_output_dim, obs_act_input_dim, obs_act_output_dim, final_input_dim, final_output_dim, num_agents, num_actions, device):
		super(TransformerCritic, self).__init__()
		
		self.name = "TransformerCritic"

		self.num_agents = num_agents
		self.num_actions = num_actions
		self.device = device

		self.CNN = nn.Sequential(
			nn.Conv2d(in_channels, 16, kernel_size=3, stride=1, padding=0, bias=True),
			nn.LeakyReLU(),
			nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=0, bias=True),
			nn.LeakyReLU()
			)

		self.state_act_embed_attn = nn.Sequential(
			nn.Linear(32+obs_act_input_dim, 128), 
			nn.LeakyReLU()
			)
		self.state_embed_attn = nn.Sequential(
			nn.Linear(32+obs_input_dim, 128), 
			nn.LeakyReLU()
			)
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
		self.state_embed_q = nn.Sequential(nn.Linear(32+obs_input_dim, obs_output_dim), nn.LeakyReLU())
		# FCN FINAL LAYER TO GET VALUES
		self.final_value_layers = nn.Sequential(
			nn.Linear(final_input_dim, 64, bias=True),
			nn.LeakyReLU(),
			nn.Linear(64, final_output_dim, bias=True)
			)
		# ********************************************************************************************************
		self.reset_parameters()


	def reset_parameters(self):
		"""Reinitialize learnable parameters."""
		gain_leaky_relu = nn.init.calculate_gain('leaky_relu')

		nn.init.orthogonal_(self.CNN[0].weight, gain=gain_leaky_relu)
		nn.init.orthogonal_(self.CNN[2].weight, gain=gain_leaky_relu)

		nn.init.xavier_uniform_(self.state_act_embed_attn[0].weight, gain=gain_leaky_relu)
		nn.init.xavier_uniform_(self.state_embed_attn[0].weight, gain=gain_leaky_relu)
		nn.init.xavier_uniform_(self.state_embed_q[0].weight, gain=gain_leaky_relu)

		nn.init.xavier_uniform_(self.key_layer.weight)
		nn.init.xavier_uniform_(self.query_layer.weight)
		nn.init.xavier_uniform_(self.attention_value_layer.weight)

		nn.init.xavier_uniform_(self.final_value_layers[0].weight, gain=gain_leaky_relu)
		nn.init.xavier_uniform_(self.final_value_layers[2].weight, gain=gain_leaky_relu)


	def forward(self, agent_states, agent_global_positions, agent_one_hot_encoding, policies, actions):
		agent_pose_id = torch.cat([agent_global_positions, agent_one_hot_encoding], dim=-1).to(self.device)
		agent_states_shape = agent_states.shape
		states = self.CNN(agent_states.reshape(-1,agent_states_shape[-3],agent_states_shape[-2],agent_states_shape[-1]))
		states = torch.cat([states.reshape(agent_states_shape[0], agent_states_shape[1],-1), agent_pose_id], dim=-1).to(self.device)
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