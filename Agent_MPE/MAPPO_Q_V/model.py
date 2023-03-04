import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from utils import gumbel_sigmoid


class MLP_Policy(nn.Module):
	def __init__(self, obs_input_dim, num_actions, num_agents, device):
		super(MLP_Policy, self).__init__()

		self.name = "MLP Policy"

		self.num_agents = num_agents
		self.num_actions = num_actions
		self.device = device
		self.Policy_MLP = nn.Sequential(
			nn.Linear(obs_input_dim, 128),
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

		nn.init.xavier_uniform_(self.Policy_MLP[0].weight)
		nn.init.xavier_uniform_(self.Policy_MLP[2].weight)
		nn.init.xavier_uniform_(self.Policy_MLP[4].weight)


	def forward(self, local_observations):
		return self.Policy_MLP(local_observations)


class Q_V_network(nn.Module):
	def __init__(self, obs_input_dim, num_heads, num_agents, num_actions, device, enable_hard_attention):
		super(Q_V_network, self).__init__()
		
		self.num_heads = num_heads
		self.num_agents = num_agents
		self.num_actions = num_actions
		self.device = device
		self.enable_hard_attention = enable_hard_attention

		# Embedding Networks
		self.state_embed = nn.Sequential(
			nn.Linear(obs_input_dim, 64, bias=True), 
			nn.GELU(),
			)
		self.state_act_embed = nn.Sequential(
			nn.Linear(obs_input_dim+self.num_actions, 64, bias=True), 
			nn.GELU(),
			)

		# Key, Query, Attention Value, Hard Attention Networks
		assert 64%self.num_heads == 0
		self.key = nn.ModuleList([nn.Sequential(
					nn.Linear(64, 64, bias=True), 
					nn.GELU()
					).to(self.device) for _ in range(self.num_heads)])
		self.query = nn.ModuleList([nn.Sequential(
					nn.Linear(64, 64, bias=True), 
					nn.GELU()
					).to(self.device) for _ in range(self.num_heads)])
		self.attention_value = nn.ModuleList([nn.Sequential(
					nn.Linear(64, 64//self.num_heads, bias=True), 
					nn.GELU()
					).to(self.device) for _ in range(self.num_heads)])

		self.attention_value_layer_norm = nn.LayerNorm(64)

		self.attention_value_linear = nn.Sequential(
			nn.Linear(64, 64),
			nn.GELU(),
			)

		self.attention_value_linear_layer_norm = nn.LayerNorm(64)

		if self.enable_hard_attention:
			self.hard_attention = nn.ModuleList([nn.Sequential(
						nn.Linear(64*2, 64//self.num_heads),
						nn.GELU(),
						).to(self.device) for _ in range(self.num_heads)])

			self.hard_attention_linear = nn.Sequential(
				nn.Linear(64, 2)
				)


		# dimesion of key
		self.d_k = 64

		# FCN FINAL LAYER TO GET Q-VALUES
		self.common_layer = nn.Sequential(
			nn.Linear(64*2, 64, bias=True), 
			nn.GELU(),
			)
		self.RNN = nn.GRUCell(64, 64)
		self.q_value_layer = nn.Sequential(
			nn.Linear(64, 64, bias=True),
			nn.GELU(),
			nn.Linear(64, self.num_actions)
			)

		self.v_value_layer = nn.Sequential(
			nn.Linear(64, 64, bias=True),
			nn.GELU(),
			nn.Linear(64, 1)
			)
			
		# ********************************************************************************************************
		self.reset_parameters()


	def reset_parameters(self):
		"""Reinitialize learnable parameters."""
		# gain = nn.init.calculate_gain('tanh', 0.01)

		# Embedding Networks
		nn.init.xavier_uniform_(self.state_embed[0].weight)
		nn.init.xavier_uniform_(self.state_act_embed[0].weight)

		# Key, Query, Attention Value, Hard Attention Networks
		for i in range(self.num_heads):
			nn.init.xavier_uniform_(self.key[i][0].weight)
			nn.init.xavier_uniform_(self.query[i][0].weight)
			nn.init.xavier_uniform_(self.attention_value[i][0].weight)
			if self.enable_hard_attention:
				nn.init.xavier_uniform_(self.hard_attention[i][0].weight)

		nn.init.xavier_uniform_(self.attention_value_linear[0].weight)
		if self.enable_hard_attention:
			nn.init.xavier_uniform_(self.hard_attention_linear[0].weight)

		nn.init.xavier_uniform_(self.common_layer[0].weight)
		nn.init.xavier_uniform_(self.q_value_layer[0].weight)
		nn.init.xavier_uniform_(self.q_value_layer[2].weight)
		nn.init.xavier_uniform_(self.v_value_layer[0].weight)
		nn.init.xavier_uniform_(self.v_value_layer[2].weight)


	# We assume that the agent in question's actions always impact its rewards
	def remove_self_loops(self, states_key):
		ret_states_keys = torch.zeros(states_key.shape[0],self.num_agents,self.num_agents-1,states_key.shape[-1])
		for i in range(self.num_agents):
			if i == 0:
				red_state = states_key[:,i,i+1:]
			elif i == self.num_agents-1:
				red_state = states_key[:,i,:i]
			else:
				red_state = torch.cat([states_key[:,i,:i],states_key[:,i,i+1:]], dim=-2)

			ret_states_keys[:,i] = red_state

		return ret_states_keys.to(self.device)

	# Setting weight value as 1 for the diagonal elements in the weight matrix
	def weight_assignment(self,weights):
		weights_new = torch.zeros(weights.shape[0], self.num_heads, self.num_agents, self.num_agents).to(self.device)
		one = torch.ones(weights.shape[0], self.num_heads, 1).to(self.device)
		for i in range(self.num_agents):
			if i == 0:
				weight_vec = torch.cat([one,weights[:,:,i,:]], dim=-1)
			elif i == self.num_agents-1:
				weight_vec = torch.cat([weights[:,:,i,:],one], dim=-1)
			else:
				weight_vec = torch.cat([weights[:,:,i,:i],one,weights[:,:,i,i:]], dim=-1)
			weights_new[:,:,i,:] = weight_vec

		return weights_new.to(self.device)


	def forward(self, states, history, actions):
		states_query = states.unsqueeze(-2) # Batch_size, Num agents, sequence_length=1, dim
		# print(states_query.shape)
		states_key = states.unsqueeze(1).repeat(1,self.num_agents,1,1) # Batch_size, Num agents, Num Agents, dim
		# print(states_key.shape)
		actions_ = actions.unsqueeze(1).repeat(1,self.num_agents,1,1) # Batch_size, Num agents, Num_Agents, dim
		# print(actions_.shape)

		states_key = self.remove_self_loops(states_key) # Batch_size, Num agents, Num Agents - 1, dim
		# print(states_key.shape)
		actions_ = self.remove_self_loops(actions_) # Batch_size, Num agents, Num Agents - 1, dim
		# print(actions_.shape)

		obs_actions = torch.cat([states_key,actions_],dim=-1).to(self.device) # Batch_size, Num agents, Num Agents - 1, dim
		# print(obs_actions.shape)

		# EMBED STATES QUERY
		states_query_embed = self.state_embed(states_query) # Batch_size, Num agents, 1, dim
		# print(states_query_embed.shape)
		# EMBED STATES QUERY
		states_key_embed = self.state_embed(states_key) # Batch_size, Num agents, Num Agents - 1, dim
		# print(states_key_embed.shape)
		# KEYS
		key_obs = torch.stack([self.key[i](states_key_embed) for i in range(self.num_heads)], dim=0).permute(1,0,2,3,4).to(self.device) # Batch_size, Num Heads, Num agents, Num Agents - 1, dim
		# print(key_obs.shape)
		# QUERIES
		query_obs = torch.stack([self.query[i](states_query_embed) for i in range(self.num_heads)], dim=0).permute(1,0,2,3,4).to(self.device) # Batch_size, Num Heads, Num agents, 1, dim
		# print(query_obs.shape)
		# HARD ATTENTION
		if self.enable_hard_attention:
			query_key_concat = torch.cat([query_obs.repeat(1,1,1,self.num_agents-1,1), key_obs], dim=-1) # Batch_size, Num Heads, Num agents, Num Agents - 1, dim
			# print(query_key_concat.shape)
			query_key_concat_intermediate = torch.cat([self.hard_attention[i](query_key_concat[:,i]) for i in range(self.num_heads)], dim=-1) # Batch_size, Num agents, Num agents-1, dim
			# print(query_key_concat_intermediate.shape)
			# GUMBEL SIGMOID, did not work that well
			# hard_attention_weights = gumbel_sigmoid(self.hard_attention_linear(query_key_concat_intermediate), hard=True) # Batch_size, Num agents, Num Agents - 1, 1
			# GUMBEL SOFTMAX
			hard_attention_weights = F.gumbel_softmax(self.hard_attention_linear(query_key_concat_intermediate), hard=True, tau=0.01)[:,:,:,1].unsqueeze(-1) # Batch_size, Num agents, Num Agents - 1, 1
			# print(hard_attention_weights.shape)
		else:
			hard_attention_weights = torch.ones(states.shape[0], self.num_agents, self.num_agents-1, 1).to(self.device)
			# print(hard_attention_weights.shape)
		# SOFT ATTENTION
		score = torch.matmul(query_obs,(key_obs*hard_attention_weights.unsqueeze(1)).transpose(-2,-1))/math.sqrt(self.d_k) # Batch_size, Num Heads, Num agents, 1, Num Agents - 1
		# print(score.shape)
		weight = F.softmax(score ,dim=-1) # Batch_size, Num Heads, Num agents, 1, Num Agents - 1
		# print(weight.shape)
		weights = self.weight_assignment(weight.squeeze(-2)) # Batch_size, Num Heads, Num agents, Num agents
		# print(weights.shape)

		# EMBED STATE ACTION POLICY
		obs_actions_embed = self.state_act_embed(obs_actions) # Batch_size, Num agents, Num agents - 1, dim
		# print(obs_actions_embed.shape)
		attention_values = torch.stack([self.attention_value[i](obs_actions_embed) for i in range(self.num_heads)], dim=0).permute(1,0,2,3,4) # Batch_size, Num heads, Num agents, Num agents - 1, dim//num_heads
		# print(attention_values.shape)
		aggregated_node_features = torch.matmul(weight, attention_values).squeeze(-2) # Batch_size, Num heads, Num agents, dim//num_heads
		# print(aggregated_node_features.shape)
		aggregated_node_features = aggregated_node_features.permute(0,2,1,3).reshape(states.shape[0], self.num_agents, -1) # Batch_size, Num agents, dim
		# print(aggregated_node_features.shape)
		aggregated_node_features_ = self.attention_value_layer_norm(torch.sum(obs_actions_embed, dim=-2)+aggregated_node_features) # Batch_size, Num agents, dim
		# print(aggregated_node_features_.shape)
		aggregated_node_features = self.attention_value_linear(aggregated_node_features_) # Batch_size, Num agents, dim
		# print(aggregated_node_features.shape)
		aggregated_node_features = self.attention_value_linear_layer_norm(aggregated_node_features_+aggregated_node_features) # Batch_size, Num agents, dim
		# print(aggregated_node_features.shape)

		curr_agent_node_features = torch.cat([states_query_embed.squeeze(-2), aggregated_node_features], dim=-1) # Batch_size, Num agents, dim
		# print(curr_agent_node_features.shape)

		curr_agent_node_features = self.common_layer(curr_agent_node_features) # Batch_size, Num agents, dim
		# print(curr_agent_node_features.shape)
		curr_agent_node_features = self.RNN(curr_agent_node_features.reshape(-1, curr_agent_node_features.shape[-1]), history.reshape(-1, curr_agent_node_features.shape[-1])).reshape(states.shape[0], self.num_agents, -1) # Batch_size, Num agents, dim
		# print(curr_agent_node_features.shape)
		Q_value = self.q_value_layer(curr_agent_node_features) # Batch_size, Num agents, num_actions
		# print(Q_value.shape)
		Q_value = torch.sum(actions*Q_value, dim=-1).unsqueeze(-1) # Batch_size, Num agents, 1
		# print(Q_value.shape)
		V_value = self.v_value_layer(curr_agent_node_features) # Batch_size, Num agents, 1
		# print(V_value.shape)

		return Q_value.squeeze(-1), V_value.squeeze(-1), curr_agent_node_features, weights


class Q_network(nn.Module):
	def __init__(self, obs_input_dim, num_heads, num_agents, num_actions, device, enable_hard_attention):
		super(Q_network, self).__init__()
		
		self.num_heads = num_heads
		self.num_agents = num_agents
		self.num_actions = num_actions
		self.device = device
		self.enable_hard_attention = enable_hard_attention

		# Embedding Networks
		self.state_embed = nn.Sequential(
			nn.Linear(obs_input_dim, 64, bias=True), 
			nn.GELU(),
			)
		self.state_act_embed = nn.Sequential(
			nn.Linear(obs_input_dim+self.num_actions, 64, bias=True), 
			nn.GELU(),
			)

		# Key, Query, Attention Value, Hard Attention Networks
		assert 64%self.num_heads == 0
		self.key = nn.ModuleList([nn.Sequential(
					nn.Linear(64, 64, bias=True), 
					nn.GELU()
					).to(self.device) for _ in range(self.num_heads)])
		self.query = nn.ModuleList([nn.Sequential(
					nn.Linear(64, 64, bias=True), 
					nn.GELU()
					).to(self.device) for _ in range(self.num_heads)])
		self.attention_value = nn.ModuleList([nn.Sequential(
					nn.Linear(64, 64//self.num_heads, bias=True), 
					nn.GELU()
					).to(self.device) for _ in range(self.num_heads)])

		self.attention_value_layer_norm = nn.LayerNorm(64)

		self.attention_value_linear = nn.Sequential(
			nn.Linear(64, 64),
			nn.GELU(),
			)

		self.attention_value_linear_layer_norm = nn.LayerNorm(64)

		if self.enable_hard_attention:
			self.hard_attention = nn.ModuleList([nn.Sequential(
						nn.Linear(64*2, 64//self.num_heads),
						nn.GELU(),
						).to(self.device) for _ in range(self.num_heads)])

			self.hard_attention_linear = nn.Sequential(
				nn.Linear(64, 2)
				)


		# dimesion of key
		self.d_k = 64

		# FCN FINAL LAYER TO GET Q-VALUES
		self.common_layer = nn.Sequential(
			nn.Linear(64*2, 64, bias=True), 
			nn.GELU(),
			)
		self.RNN = nn.GRUCell(64, 64)
		self.q_value_layer = nn.Sequential(
			nn.Linear(64, 64, bias=True),
			nn.GELU(),
			nn.Linear(64, self.num_actions)
			)
			
		# ********************************************************************************************************
		self.reset_parameters()


	def reset_parameters(self):
		"""Reinitialize learnable parameters."""
		# gain = nn.init.calculate_gain('tanh', 0.01)

		# Embedding Networks
		nn.init.xavier_uniform_(self.state_embed[0].weight)
		nn.init.xavier_uniform_(self.state_act_embed[0].weight)

		# Key, Query, Attention Value, Hard Attention Networks
		for i in range(self.num_heads):
			nn.init.xavier_uniform_(self.key[i][0].weight)
			nn.init.xavier_uniform_(self.query[i][0].weight)
			nn.init.xavier_uniform_(self.attention_value[i][0].weight)
			if self.enable_hard_attention:
				nn.init.xavier_uniform_(self.hard_attention[i][0].weight)

		nn.init.xavier_uniform_(self.attention_value_linear[0].weight)
		if self.enable_hard_attention:
			nn.init.xavier_uniform_(self.hard_attention_linear[0].weight)

		nn.init.xavier_uniform_(self.common_layer[0].weight)
		nn.init.xavier_uniform_(self.q_value_layer[0].weight)
		nn.init.xavier_uniform_(self.q_value_layer[2].weight)


	# We assume that the agent in question's actions always impact its rewards
	def remove_self_loops(self, states_key):
		ret_states_keys = torch.zeros(states_key.shape[0],self.num_agents,self.num_agents-1,states_key.shape[-1])
		for i in range(self.num_agents):
			if i == 0:
				red_state = states_key[:,i,i+1:]
			elif i == self.num_agents-1:
				red_state = states_key[:,i,:i]
			else:
				red_state = torch.cat([states_key[:,i,:i],states_key[:,i,i+1:]], dim=-2)

			ret_states_keys[:,i] = red_state

		return ret_states_keys.to(self.device)

	# Setting weight value as 1 for the diagonal elements in the weight matrix
	def weight_assignment(self,weights):
		weights_new = torch.zeros(weights.shape[0], self.num_heads, self.num_agents, self.num_agents).to(self.device)
		one = torch.ones(weights.shape[0], self.num_heads, 1).to(self.device)
		for i in range(self.num_agents):
			if i == 0:
				weight_vec = torch.cat([one,weights[:,:,i,:]], dim=-1)
			elif i == self.num_agents-1:
				weight_vec = torch.cat([weights[:,:,i,:],one], dim=-1)
			else:
				weight_vec = torch.cat([weights[:,:,i,:i],one,weights[:,:,i,i:]], dim=-1)
			weights_new[:,:,i,:] = weight_vec

		return weights_new.to(self.device)


	def forward(self, states, history, actions):
		states_query = states.unsqueeze(-2) # Batch_size, Num agents, sequence_length=1, dim
		# print(states_query.shape)
		states_key = states.unsqueeze(1).repeat(1,self.num_agents,1,1) # Batch_size, Num agents, Num Agents, dim
		# print(states_key.shape)
		actions_ = actions.unsqueeze(1).repeat(1,self.num_agents,1,1) # Batch_size, Num agents, Num_Agents, dim
		# print(actions_.shape)

		states_key = self.remove_self_loops(states_key) # Batch_size, Num agents, Num Agents - 1, dim
		# print(states_key.shape)
		actions_ = self.remove_self_loops(actions_) # Batch_size, Num agents, Num Agents - 1, dim
		# print(actions_.shape)

		obs_actions = torch.cat([states_key,actions_],dim=-1).to(self.device) # Batch_size, Num agents, Num Agents - 1, dim
		# print(obs_actions.shape)

		# EMBED STATES QUERY
		states_query_embed = self.state_embed(states_query) # Batch_size, Num agents, 1, dim
		# print(states_query_embed.shape)
		# EMBED STATES QUERY
		states_key_embed = self.state_embed(states_key) # Batch_size, Num agents, Num Agents - 1, dim
		# print(states_key_embed.shape)
		# KEYS
		key_obs = torch.stack([self.key[i](states_key_embed) for i in range(self.num_heads)], dim=0).permute(1,0,2,3,4).to(self.device) # Batch_size, Num Heads, Num agents, Num Agents - 1, dim
		# print(key_obs.shape)
		# QUERIES
		query_obs = torch.stack([self.query[i](states_query_embed) for i in range(self.num_heads)], dim=0).permute(1,0,2,3,4).to(self.device) # Batch_size, Num Heads, Num agents, 1, dim
		# print(query_obs.shape)
		# HARD ATTENTION
		if self.enable_hard_attention:
			query_key_concat = torch.cat([query_obs.repeat(1,1,1,self.num_agents-1,1), key_obs], dim=-1) # Batch_size, Num Heads, Num agents, Num Agents - 1, dim
			# print(query_key_concat.shape)
			query_key_concat_intermediate = torch.cat([self.hard_attention[i](query_key_concat[:,i]) for i in range(self.num_heads)], dim=-1) # Batch_size, Num agents, Num agents-1, dim
			# print(query_key_concat_intermediate.shape)
			# GUMBEL SIGMOID, did not work that well
			# hard_attention_weights = gumbel_sigmoid(self.hard_attention_linear(query_key_concat_intermediate), hard=True) # Batch_size, Num agents, Num Agents - 1, 1
			# GUMBEL SOFTMAX
			hard_attention_weights = F.gumbel_softmax(self.hard_attention_linear(query_key_concat_intermediate), hard=True, tau=0.01)[:,:,:,1].unsqueeze(-1) # Batch_size, Num agents, Num Agents - 1, 1
			# print(hard_attention_weights.shape)
		else:
			hard_attention_weights = torch.ones(states.shape[0], self.num_agents, self.num_agents-1, 1).to(self.device)
			# print(hard_attention_weights.shape)
		# SOFT ATTENTION
		score = torch.matmul(query_obs,(key_obs*hard_attention_weights.unsqueeze(1)).transpose(-2,-1))/math.sqrt(self.d_k) # Batch_size, Num Heads, Num agents, 1, Num Agents - 1
		# print(score.shape)
		weight = F.softmax(score ,dim=-1) # Batch_size, Num Heads, Num agents, 1, Num Agents - 1
		# print(weight.shape)
		weights = self.weight_assignment(weight.squeeze(-2)) # Batch_size, Num Heads, Num agents, Num agents
		# print(weights.shape)

		# EMBED STATE ACTION POLICY
		obs_actions_embed = self.state_act_embed(obs_actions) # Batch_size, Num agents, Num agents - 1, dim
		# print(obs_actions_embed.shape)
		attention_values = torch.stack([self.attention_value[i](obs_actions_embed) for i in range(self.num_heads)], dim=0).permute(1,0,2,3,4) # Batch_size, Num heads, Num agents, Num agents - 1, dim//num_heads
		# print(attention_values.shape)
		aggregated_node_features = torch.matmul(weight, attention_values).squeeze(-2) # Batch_size, Num heads, Num agents, dim//num_heads
		# print(aggregated_node_features.shape)
		aggregated_node_features = aggregated_node_features.permute(0,2,1,3).reshape(states.shape[0], self.num_agents, -1) # Batch_size, Num agents, dim
		# print(aggregated_node_features.shape)
		aggregated_node_features_ = self.attention_value_layer_norm(torch.sum(obs_actions_embed, dim=-2)+aggregated_node_features) # Batch_size, Num agents, dim
		# print(aggregated_node_features_.shape)
		aggregated_node_features = self.attention_value_linear(aggregated_node_features_) # Batch_size, Num agents, dim
		# print(aggregated_node_features.shape)
		aggregated_node_features = self.attention_value_linear_layer_norm(aggregated_node_features_+aggregated_node_features) # Batch_size, Num agents, dim
		# print(aggregated_node_features.shape)

		curr_agent_node_features = torch.cat([states_query_embed.squeeze(-2), aggregated_node_features], dim=-1) # Batch_size, Num agents, dim
		# print(curr_agent_node_features.shape)

		curr_agent_node_features = self.common_layer(curr_agent_node_features) # Batch_size, Num agents, dim
		# print(curr_agent_node_features.shape)
		curr_agent_node_features = self.RNN(curr_agent_node_features.reshape(-1, curr_agent_node_features.shape[-1]), history.reshape(-1, curr_agent_node_features.shape[-1])).reshape(states.shape[0], self.num_agents, -1) # Batch_size, Num agents, dim
		# print(curr_agent_node_features.shape)
		Q_value = self.q_value_layer(curr_agent_node_features) # Batch_size, Num agents, num_actions
		# print(Q_value.shape)
		Q_value = torch.sum(actions*Q_value, dim=-1).unsqueeze(-1) # Batch_size, Num agents, 1
		# print(Q_value.shape)

		return Q_value.squeeze(-1), curr_agent_node_features, weights


class V_network(nn.Module):
	def __init__(self, obs_input_dim, num_heads, num_agents, num_actions, device, enable_hard_attention):
		super(V_network, self).__init__()
		
		self.num_heads = num_heads
		self.num_agents = num_agents
		self.num_actions = num_actions
		self.device = device
		self.enable_hard_attention = enable_hard_attention

		# Embedding Networks
		self.state_embed = nn.Sequential(
			nn.Linear(obs_input_dim, 64, bias=True), 
			nn.GELU(),
			)
		self.state_act_embed = nn.Sequential(
			nn.Linear(obs_input_dim+self.num_actions, 64, bias=True), 
			nn.GELU(),
			)

		# Key, Query, Attention Value, Hard Attention Networks
		assert 64%self.num_heads == 0
		self.key = nn.ModuleList([nn.Sequential(
					nn.Linear(64, 64, bias=True), 
					nn.GELU()
					).to(self.device) for _ in range(self.num_heads)])
		self.query = nn.ModuleList([nn.Sequential(
					nn.Linear(64, 64, bias=True), 
					nn.GELU()
					).to(self.device) for _ in range(self.num_heads)])
		self.attention_value = nn.ModuleList([nn.Sequential(
					nn.Linear(64, 64//self.num_heads, bias=True), 
					nn.GELU()
					).to(self.device) for _ in range(self.num_heads)])

		self.attention_value_layer_norm = nn.LayerNorm(64)

		self.attention_value_linear = nn.Sequential(
			nn.Linear(64, 64),
			nn.GELU(),
			)

		self.attention_value_linear_layer_norm = nn.LayerNorm(64)

		if self.enable_hard_attention:
			self.hard_attention = nn.ModuleList([nn.Sequential(
						nn.Linear(64*2, 64//self.num_heads),
						nn.GELU(),
						).to(self.device) for _ in range(self.num_heads)])

			self.hard_attention_linear = nn.Sequential(
				nn.Linear(64, 2)
				)


		# dimesion of key
		self.d_k = 64

		# FCN FINAL LAYER TO GET Q-VALUES
		self.common_layer = nn.Sequential(
			nn.Linear(64*2, 64, bias=True), 
			nn.GELU(),
			)
		self.RNN = nn.GRUCell(64, 64)

		self.v_value_layer = nn.Sequential(
			nn.Linear(64, 64, bias=True),
			nn.GELU(),
			nn.Linear(64, 1)
			)
			
		# ********************************************************************************************************
		self.reset_parameters()


	def reset_parameters(self):
		"""Reinitialize learnable parameters."""
		# gain = nn.init.calculate_gain('tanh', 0.01)

		# Embedding Networks
		nn.init.xavier_uniform_(self.state_embed[0].weight)
		nn.init.xavier_uniform_(self.state_act_embed[0].weight)

		# Key, Query, Attention Value, Hard Attention Networks
		for i in range(self.num_heads):
			nn.init.xavier_uniform_(self.key[i][0].weight)
			nn.init.xavier_uniform_(self.query[i][0].weight)
			nn.init.xavier_uniform_(self.attention_value[i][0].weight)
			if self.enable_hard_attention:
				nn.init.xavier_uniform_(self.hard_attention[i][0].weight)

		nn.init.xavier_uniform_(self.attention_value_linear[0].weight)
		if self.enable_hard_attention:
			nn.init.xavier_uniform_(self.hard_attention_linear[0].weight)

		nn.init.xavier_uniform_(self.common_layer[0].weight)
		nn.init.xavier_uniform_(self.v_value_layer[0].weight)
		nn.init.xavier_uniform_(self.v_value_layer[2].weight)


	# We assume that the agent in question's actions always impact its rewards
	def remove_self_loops(self, states_key):
		ret_states_keys = torch.zeros(states_key.shape[0],self.num_agents,self.num_agents-1,states_key.shape[-1])
		for i in range(self.num_agents):
			if i == 0:
				red_state = states_key[:,i,i+1:]
			elif i == self.num_agents-1:
				red_state = states_key[:,i,:i]
			else:
				red_state = torch.cat([states_key[:,i,:i],states_key[:,i,i+1:]], dim=-2)

			ret_states_keys[:,i] = red_state

		return ret_states_keys.to(self.device)

	# Setting weight value as 1 for the diagonal elements in the weight matrix
	def weight_assignment(self,weights):
		weights_new = torch.zeros(weights.shape[0], self.num_heads, self.num_agents, self.num_agents).to(self.device)
		one = torch.ones(weights.shape[0], self.num_heads, 1).to(self.device)
		for i in range(self.num_agents):
			if i == 0:
				weight_vec = torch.cat([one,weights[:,:,i,:]], dim=-1)
			elif i == self.num_agents-1:
				weight_vec = torch.cat([weights[:,:,i,:],one], dim=-1)
			else:
				weight_vec = torch.cat([weights[:,:,i,:i],one,weights[:,:,i,i:]], dim=-1)
			weights_new[:,:,i,:] = weight_vec

		return weights_new.to(self.device)


	def forward(self, states, history, actions):
		states_query = states.unsqueeze(-2) # Batch_size, Num agents, sequence_length=1, dim
		# print(states_query.shape)
		states_key = states.unsqueeze(1).repeat(1,self.num_agents,1,1) # Batch_size, Num agents, Num Agents, dim
		# print(states_key.shape)
		actions_ = actions.unsqueeze(1).repeat(1,self.num_agents,1,1) # Batch_size, Num agents, Num_Agents, dim
		# print(actions_.shape)

		states_key = self.remove_self_loops(states_key) # Batch_size, Num agents, Num Agents - 1, dim
		# print(states_key.shape)
		actions_ = self.remove_self_loops(actions_) # Batch_size, Num agents, Num Agents - 1, dim
		# print(actions_.shape)

		obs_actions = torch.cat([states_key,actions_],dim=-1).to(self.device) # Batch_size, Num agents, Num Agents - 1, dim
		# print(obs_actions.shape)

		# EMBED STATES QUERY
		states_query_embed = self.state_embed(states_query) # Batch_size, Num agents, 1, dim
		# print(states_query_embed.shape)
		# EMBED STATES QUERY
		states_key_embed = self.state_embed(states_key) # Batch_size, Num agents, Num Agents - 1, dim
		# print(states_key_embed.shape)
		# KEYS
		key_obs = torch.stack([self.key[i](states_key_embed) for i in range(self.num_heads)], dim=0).permute(1,0,2,3,4).to(self.device) # Batch_size, Num Heads, Num agents, Num Agents - 1, dim
		# print(key_obs.shape)
		# QUERIES
		query_obs = torch.stack([self.query[i](states_query_embed) for i in range(self.num_heads)], dim=0).permute(1,0,2,3,4).to(self.device) # Batch_size, Num Heads, Num agents, 1, dim
		# print(query_obs.shape)
		# HARD ATTENTION
		if self.enable_hard_attention:
			query_key_concat = torch.cat([query_obs.repeat(1,1,1,self.num_agents-1,1), key_obs], dim=-1) # Batch_size, Num Heads, Num agents, Num Agents - 1, dim
			# print(query_key_concat.shape)
			query_key_concat_intermediate = torch.cat([self.hard_attention[i](query_key_concat[:,i]) for i in range(self.num_heads)], dim=-1) # Batch_size, Num agents, Num agents-1, dim
			# print(query_key_concat_intermediate.shape)
			# GUMBEL SIGMOID, did not work that well
			# hard_attention_weights = gumbel_sigmoid(self.hard_attention_linear(query_key_concat_intermediate), hard=True) # Batch_size, Num agents, Num Agents - 1, 1
			# GUMBEL SOFTMAX
			hard_attention_weights = F.gumbel_softmax(self.hard_attention_linear(query_key_concat_intermediate), hard=True, tau=0.01)[:,:,:,1].unsqueeze(-1) # Batch_size, Num agents, Num Agents - 1, 1
			# print(hard_attention_weights.shape)
		else:
			hard_attention_weights = torch.ones(states.shape[0], self.num_agents, self.num_agents-1, 1).to(self.device)
			# print(hard_attention_weights.shape)
		# SOFT ATTENTION
		score = torch.matmul(query_obs,(key_obs*hard_attention_weights.unsqueeze(1)).transpose(-2,-1))/math.sqrt(self.d_k) # Batch_size, Num Heads, Num agents, 1, Num Agents - 1
		# print(score.shape)
		weight = F.softmax(score ,dim=-1) # Batch_size, Num Heads, Num agents, 1, Num Agents - 1
		# print(weight.shape)
		weights = self.weight_assignment(weight.squeeze(-2)) # Batch_size, Num Heads, Num agents, Num agents
		# print(weights.shape)

		# EMBED STATE ACTION POLICY
		obs_actions_embed = self.state_act_embed(obs_actions) # Batch_size, Num agents, Num agents - 1, dim
		# print(obs_actions_embed.shape)
		attention_values = torch.stack([self.attention_value[i](obs_actions_embed) for i in range(self.num_heads)], dim=0).permute(1,0,2,3,4) # Batch_size, Num heads, Num agents, Num agents - 1, dim//num_heads
		# print(attention_values.shape)
		aggregated_node_features = torch.matmul(weight, attention_values).squeeze(-2) # Batch_size, Num heads, Num agents, dim//num_heads
		# print(aggregated_node_features.shape)
		aggregated_node_features = aggregated_node_features.permute(0,2,1,3).reshape(states.shape[0], self.num_agents, -1) # Batch_size, Num agents, dim
		# print(aggregated_node_features.shape)
		aggregated_node_features_ = self.attention_value_layer_norm(torch.sum(obs_actions_embed, dim=-2)+aggregated_node_features) # Batch_size, Num agents, dim
		# print(aggregated_node_features_.shape)
		aggregated_node_features = self.attention_value_linear(aggregated_node_features_) # Batch_size, Num agents, dim
		# print(aggregated_node_features.shape)
		aggregated_node_features = self.attention_value_linear_layer_norm(aggregated_node_features_+aggregated_node_features) # Batch_size, Num agents, dim
		# print(aggregated_node_features.shape)

		curr_agent_node_features = torch.cat([states_query_embed.squeeze(-2), aggregated_node_features], dim=-1) # Batch_size, Num agents, dim
		# print(curr_agent_node_features.shape)

		curr_agent_node_features = self.common_layer(curr_agent_node_features) # Batch_size, Num agents, dim
		# print(curr_agent_node_features.shape)
		curr_agent_node_features = self.RNN(curr_agent_node_features.reshape(-1, curr_agent_node_features.shape[-1]), history.reshape(-1, curr_agent_node_features.shape[-1])).reshape(states.shape[0], self.num_agents, -1) # Batch_size, Num agents, dim
		# print(curr_agent_node_features.shape)
		V_value = self.v_value_layer(curr_agent_node_features) # Batch_size, Num agents, 1
		# print(V_value.shape)

		return V_value.squeeze(-1), curr_agent_node_features, weights
