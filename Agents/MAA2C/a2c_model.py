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

		self.state_embed = nn.Sequential(
			nn.Linear(32+obs_input_dim, 128, bias=True), 
			nn.Tanh()
			)
		self.key_layer = nn.Linear(128, obs_output_dim, bias=True)
		self.query_layer = nn.Linear(128, obs_output_dim, bias=True)
		
		self.state_act_pol_embed = nn.Sequential(
			nn.Linear(32+obs_act_input_dim, obs_act_output_dim, bias=True), 
			nn.Tanh()
			)
		self.attention_value_layer = nn.Sequential(
			nn.Linear(obs_act_output_dim, 128, bias=True), 
			nn.Tanh()
			)

		# dimesion of key
		self.d_k_obs_act = obs_output_dim  

		# ********************************************************************************************************

		# ********************************************************************************************************
		# FCN FINAL LAYER TO GET VALUES
		self.final_value_layers = nn.Sequential(
				nn.Linear(final_input_dim, 64, bias=True), 
				nn.Tanh(),
				nn.Linear(64, final_output_dim, bias=True)
				)
		# ********************************************************************************************************	

		self.place_policies = torch.zeros(self.num_agents,self.num_agents,32+obs_act_input_dim).to(self.device)
		self.place_actions = torch.ones(self.num_agents,self.num_agents,32+obs_act_input_dim).to(self.device)
		one_hots = torch.ones(32+obs_act_input_dim)
		zero_hots = torch.zeros(32+obs_act_input_dim)

		for j in range(self.num_agents):
			self.place_policies[j][j] = one_hots
			self.place_actions[j][j] = zero_hots


		self.reset_parameters()


	def reset_parameters(self):
		"""Reinitialize learnable parameters."""
		gain_leaky_relu = nn.init.calculate_gain('leaky_relu')
		gain = nn.init.calculate_gain('tanh')

		nn.init.orthogonal_(self.CNN[0].weight, gain=gain_leaky_relu)
		nn.init.orthogonal_(self.CNN[2].weight, gain=gain_leaky_relu)

		nn.init.orthogonal_(self.state_embed[0].weight, gain=gain)
		nn.init.orthogonal_(self.state_act_pol_embed[0].weight, gain=gain)

		nn.init.orthogonal_(self.key_layer.weight)
		nn.init.orthogonal_(self.query_layer.weight)
		nn.init.orthogonal_(self.attention_value_layer[0].weight)

		nn.init.orthogonal_(self.final_value_layers[0].weight, gain=gain)
		nn.init.orthogonal_(self.final_value_layers[2].weight, gain=gain)



	def forward(self, agent_states, agent_global_positions, agent_one_hot_encoding, policies, actions):
		agent_pose_id = torch.cat([agent_global_positions, agent_one_hot_encoding], dim=-1).to(self.device)
		agent_states_shape = agent_states.shape
		states = self.CNN(agent_states.reshape(-1,agent_states_shape[-3],agent_states_shape[-2],agent_states_shape[-1]))
		states = torch.cat([states.reshape(agent_states_shape[0], agent_states_shape[1],-1), agent_pose_id], dim=-1).to(self.device)
		# EMBED STATES
		states_embed = self.state_embed(states)
		# KEYS
		key_obs = self.key_layer(states_embed)
		# QUERIES
		query_obs = self.query_layer(states_embed)
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
		attention_values = self.attention_value_layer(obs_actions_policies_embed)
		attention_values = attention_values.repeat(1,self.num_agents,1,1).reshape(attention_values.shape[0],self.num_agents,self.num_agents,self.num_agents,-1)
		
		weight = weight.unsqueeze(-2).repeat(1,1,self.num_agents,1).unsqueeze(-1)
		weighted_attention_values = attention_values*weight
		node_features = torch.sum(weighted_attention_values, dim=-2)

		Value = self.final_value_layers(node_features)

		return Value, ret_weight