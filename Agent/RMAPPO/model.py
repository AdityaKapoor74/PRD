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

		self.mask_value = torch.tensor(
				torch.finfo(torch.float).min, dtype=torch.float
			)
		self.num_agents = num_agents
		self.num_actions = num_actions
		self.device = device
		self.Layer_1 = nn.Sequential(nn.Linear(obs_input_dim, 64), nn.GELU())
		self.RNN = nn.GRU(input_size=64, hidden_size=64, num_layers=1, batch_first=True)
		self.Layer_2 = nn.Linear(64, num_actions)

		self.reset_parameters()

	def reset_parameters(self):

		nn.init.xavier_uniform_(self.Layer_1[0].weight)
		nn.init.xavier_uniform_(self.Layer_2.weight)


	def forward(self, local_observations, hidden_state, mask_actions=None, update=False):
		if update == False:
			intermediate = self.Layer_1(local_observations)
			output, h = self.RNN(intermediate, hidden_state)
			logits = self.Layer_2(output)
			logits = torch.where(mask_actions, logits, self.mask_value)
			return F.softmax(logits, dim=-1).squeeze(1), h
		else:
			# local_observations --> batch, timesteps, num_agents, dim
			batch, timesteps, num_agents, _ = local_observations.shape
			intermediate = self.Layer_1(local_observations)
			intermediate = intermediate.permute(0, 2, 1, 3).reshape(batch*num_agents, timesteps, -1)
			output, h = self.RNN(intermediate, hidden_state)
			output = output.reshape(batch, num_agents, timesteps, -1).permute(0, 2, 1, 3).reshape(batch*timesteps, num_agents, -1)
			logits = self.Layer_2(output)
			logits = torch.where(mask_actions, logits, self.mask_value)
			return F.softmax(logits, dim=-1), h



class AttentionDropout(nn.Module):
	def __init__(self, dropout_prob):
		super(AttentionDropout, self).__init__()
		self.dropout_prob = dropout_prob
	
	def forward(self, attention_scores):
		# Apply dropout to attention scores
		mask = (torch.rand_like(attention_scores) > self.dropout_prob).float()
		attention_scores = attention_scores * mask
		return attention_scores



class Q_network(nn.Module):
	def __init__(
		self, 
		ally_obs_input_dim, 
		enemy_obs_input_dim,
		num_heads, 
		num_agents, 
		num_enemies,
		num_actions, 
		device, 
		enable_hard_attention, 
		attention_dropout_prob, 
		temperature
		):
		super(Q_network, self).__init__()
		
		self.num_heads = num_heads
		self.num_agents = num_agents
		self.num_enemies = num_enemies
		self.num_actions = num_actions
		self.device = device
		self.enable_hard_attention = enable_hard_attention

		self.attention_dropout = AttentionDropout(dropout_prob=attention_dropout_prob)

		# self.positional_embedding = nn.Parameter(torch.randn(num_agents, 64))

		self.temperature = temperature

		# Embedding Networks
		self.ally_state_embed_1 = nn.Sequential(
			nn.Linear(ally_obs_input_dim, 64, bias=True), 
			nn.GELU(),
			)

		self.ally_state_embed_2 = nn.Sequential(
			nn.Linear(ally_obs_input_dim, 32, bias=True), 
			nn.GELU(),
			)

		self.enemy_state_embed = nn.Sequential(
			nn.Linear(enemy_obs_input_dim*self.num_enemies, 64, bias=True),
			nn.GELU(),
			)

		self.ally_state_act_embed = nn.Sequential(
			nn.Linear(ally_obs_input_dim+self.num_actions, 64, bias=True), 
			nn.GELU(),
			)

		# Key, Query, Attention Value, Hard Attention Networks
		assert 64%self.num_heads == 0
		self.key = nn.ModuleList([nn.Sequential(
					nn.Linear(64, 64, bias=True), 
					# nn.GELU()
					).to(self.device) for _ in range(self.num_heads)])
		self.query = nn.ModuleList([nn.Sequential(
					nn.Linear(64, 64, bias=True), 
					# nn.GELU()
					).to(self.device) for _ in range(self.num_heads)])
		self.attention_value = nn.ModuleList([nn.Sequential(
					nn.Linear(64, 64//self.num_heads, bias=True), 
					# nn.GELU()
					).to(self.device) for _ in range(self.num_heads)])

		self.attention_value_layer_norm = nn.LayerNorm(64)

		self.attention_value_linear = nn.Sequential(
			nn.Linear(64, 64),
			# nn.GELU(),
			)

		self.attention_value_linear_layer_norm = nn.LayerNorm(64)

		if self.enable_hard_attention:
			self.hard_attention = nn.ModuleList([nn.Sequential(
						nn.Linear(64*2, 64//self.num_heads),
						# nn.GELU(),
						).to(self.device) for _ in range(self.num_heads)])

			self.hard_attention_linear = nn.Sequential(
				nn.Linear(64, 2)
				)


		# dimesion of key
		self.d_k = 64

		# FCN FINAL LAYER TO GET Q-VALUES
		self.common_layer = nn.Sequential(
			nn.Linear(32+64+64, 64, bias=True), 
			nn.GELU(),
			)
		self.RNN = nn.GRU(input_size=64, hidden_size=64, num_layers=1, batch_first=True)
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
		nn.init.xavier_uniform_(self.ally_state_embed_1[0].weight)
		nn.init.xavier_uniform_(self.ally_state_embed_2[0].weight)
		nn.init.xavier_uniform_(self.enemy_state_embed[0].weight)
		nn.init.xavier_uniform_(self.ally_state_act_embed[0].weight)

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
	def weight_assignment(self, weights):
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

	def forward(self, states, enemy_states, actions, rnn_hidden_state):
		batch, timesteps, num_agents, _ = states.shape
		_, _, num_enemies, _ = enemy_states.shape
		states = states.reshape(batch*timesteps, num_agents, -1)
		enemy_states = enemy_states.reshape(batch*timesteps, num_enemies, -1)
		actions = actions.reshape(batch*timesteps, num_agents, -1)

		# EMBED STATES KEY & QUERY
		states_embed = self.ally_state_embed_1(states)
		states_query_embed = states_embed.unsqueeze(-2) # Batch size, Num Agents, 1, dim
		# print(states_query_embed.shape)
		# EMBED STATES QUERY
		states_key_embed = states_embed.unsqueeze(1).repeat(1,self.num_agents, 1, 1) # Batch_size, Num agents, Num Agents, dim
		states_key_embed = self.remove_self_loops(states_key_embed) # Batch_size, Num agents, Num Agents - 1, dim
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
			hard_attention_weights = F.gumbel_softmax(self.hard_attention_linear(query_key_concat_intermediate), hard=True, tau=1.0)[:,:,:,1].unsqueeze(-1) # Batch_size, Num agents, Num Agents - 1, 1
			# print(hard_attention_weights.shape)
		else:
			hard_attention_weights = torch.ones(states.shape[0], self.num_agents, self.num_agents-1, 1).to(self.device)
			# print(hard_attention_weights.shape)
		# SOFT ATTENTION
		score = torch.matmul(query_obs,(key_obs).transpose(-2,-1))/math.sqrt(self.d_k) # Batch_size, Num Heads, Num agents, 1, Num Agents - 1
		# print(score.shape)
		weight = F.softmax(score/self.temperature ,dim=-1)*hard_attention_weights.unsqueeze(1).permute(0, 1, 2, 4, 3) # Batch_size, Num Heads, Num agents, 1, Num Agents - 1
		# print(weight.shape)
		weights = self.weight_assignment(weight.squeeze(-2)) # Batch_size, Num Heads, Num agents, Num agents
		# print(weights.shape)

		for head in range(self.num_heads):
			weights[:, head, :, :] = self.attention_dropout(weights[:, head, :, :])

		# EMBED STATE ACTION
		obs_actions = torch.cat([states, actions], dim=-1).to(self.device) # Batch_size, Num agents, dim
		obs_actions_embed_ = self.ally_state_act_embed(obs_actions) #+ self.positional_embedding.unsqueeze(0) # Batch_size, Num agents, dim
		obs_actions_embed = self.remove_self_loops(obs_actions_embed_.unsqueeze(1).repeat(1, self.num_agents, 1, 1)) # Batch_size, Num agents, Num agents - 1, dim
		# print(obs_actions_embed.shape)
		attention_values = torch.stack([self.attention_value[i](obs_actions_embed) for i in range(self.num_heads)], dim=0).permute(1,0,2,3,4) # Batch_size, Num heads, Num agents, Num agents - 1, dim//num_heads
		# print(attention_values.shape)
		aggregated_node_features = torch.matmul(weight, attention_values).squeeze(-2) # Batch_size, Num heads, Num agents, dim//num_heads
		# print(aggregated_node_features.shape)
		aggregated_node_features = aggregated_node_features.permute(0,2,1,3).reshape(states.shape[0], self.num_agents, -1) # Batch_size, Num agents, dim
		# print(aggregated_node_features.shape)
		aggregated_node_features_ = self.attention_value_layer_norm(obs_actions_embed_+aggregated_node_features) # Batch_size, Num agents, dim
		# print(aggregated_node_features_.shape)
		aggregated_node_features = self.attention_value_linear(aggregated_node_features_) # Batch_size, Num agents, dim
		# print(aggregated_node_features.shape)
		aggregated_node_features = self.attention_value_linear_layer_norm(aggregated_node_features_+aggregated_node_features) # Batch_size, Num agents, dim
		# print(aggregated_node_features.shape)
		final_states_embed = self.ally_state_embed_2(states)
		enemy_state_embed = self.enemy_state_embed(enemy_states.reshape(enemy_states.shape[0], -1)).unsqueeze(1).repeat(1, self.num_agents, 1)

		curr_agent_node_features = torch.cat([final_states_embed, enemy_state_embed, aggregated_node_features], dim=-1) # Batch_size, Num agents, dim
		# print(curr_agent_node_features.shape)

		curr_agent_node_features = self.common_layer(curr_agent_node_features) # Batch_size, Num agents, dim
		# print(curr_agent_node_features.shape)
		# curr_agent_node_features = self.RNN(curr_agent_node_features.reshape(-1, curr_agent_node_features.shape[-1]), history.reshape(-1, curr_agent_node_features.shape[-1])).reshape(states.shape[0], self.num_agents, -1) # Batch_size, Num agents, dim
		# print(curr_agent_node_features.shape)
		curr_agent_node_features = curr_agent_node_features.reshape(batch, timesteps, num_agents, -1).permute(0, 2, 1, 3).reshape(batch*num_agents, timesteps, -1)
		output, h = self.RNN(curr_agent_node_features, rnn_hidden_state)
		output = output.reshape(batch, num_agents, timesteps, -1).permute(0, 2, 1, 3).reshape(batch*timesteps, num_agents, -1)
		Q_value = self.q_value_layer(output) # Batch_size, Num agents, num_actions
		# print(Q_value.shape)
		Q_value = torch.sum(actions*Q_value, dim=-1).unsqueeze(-1) # Batch_size, Num agents, 1
		# print(Q_value.shape)

		return Q_value.squeeze(-1), weights, score, h


class V_network(nn.Module):
	def __init__(
		self, 
		ally_obs_input_dim, 
		enemy_obs_input_dim,
		num_heads, 
		num_agents, 
		num_enemies,
		num_actions, 
		device, 
		enable_hard_attention, 
		attention_dropout_prob, 
		temperature
		):
		super(V_network, self).__init__()
		
		self.num_heads = num_heads
		self.num_agents = num_agents
		self.num_enemies = num_enemies
		self.num_actions = num_actions
		self.device = device
		self.enable_hard_attention = enable_hard_attention

		self.attention_dropout = AttentionDropout(dropout_prob=attention_dropout_prob)

		# self.positional_embedding = nn.Parameter(torch.randn(num_agents, 64))

		self.temperature = temperature

		# Embedding Networks
		self.ally_state_embed_1 = nn.Sequential(
			nn.Linear(ally_obs_input_dim, 64, bias=True), 
			nn.GELU(),
			)

		self.ally_state_embed_2 = nn.Sequential(
			nn.Linear(ally_obs_input_dim, 32, bias=True), 
			nn.GELU(),
			)

		self.enemy_state_embed = nn.Sequential(
			nn.Linear(enemy_obs_input_dim*self.num_enemies, 64, bias=True),
			nn.GELU(),
			)

		self.ally_state_act_embed = nn.Sequential(
			nn.Linear(ally_obs_input_dim+self.num_actions, 64, bias=True), 
			nn.GELU(),
			)

		# Key, Query, Attention Value, Hard Attention Networks
		assert 64%self.num_heads == 0
		self.key = nn.ModuleList([nn.Sequential(
					nn.Linear(64, 64, bias=True), 
					# nn.GELU()
					).to(self.device) for _ in range(self.num_heads)])
		self.query = nn.ModuleList([nn.Sequential(
					nn.Linear(64, 64, bias=True), 
					# nn.GELU()
					).to(self.device) for _ in range(self.num_heads)])
		self.attention_value = nn.ModuleList([nn.Sequential(
					nn.Linear(64, 64//self.num_heads, bias=True), 
					# nn.GELU()
					).to(self.device) for _ in range(self.num_heads)])

		self.attention_value_layer_norm = nn.LayerNorm(64)

		self.attention_value_linear = nn.Sequential(
			nn.Linear(64, 64),
			# nn.GELU(),
			)

		self.attention_value_linear_layer_norm = nn.LayerNorm(64)

		if self.enable_hard_attention:
			self.hard_attention = nn.ModuleList([nn.Sequential(
						nn.Linear(64*2, 64//self.num_heads),
						# nn.GELU(),
						).to(self.device) for _ in range(self.num_heads)])

			self.hard_attention_linear = nn.Sequential(
				nn.Linear(64, 2)
				)


		# dimesion of key
		self.d_k = 64

		# FCN FINAL LAYER TO GET Q-VALUES
		self.common_layer = nn.Sequential(
			nn.Linear(32+64+64, 64, bias=True), 
			nn.GELU(),
			)
		self.RNN = nn.GRU(input_size=64, hidden_size=64, num_layers=1, batch_first=True)
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
		nn.init.xavier_uniform_(self.ally_state_embed_1[0].weight)
		nn.init.xavier_uniform_(self.ally_state_embed_2[0].weight)
		nn.init.xavier_uniform_(self.enemy_state_embed[0].weight)
		nn.init.xavier_uniform_(self.ally_state_act_embed[0].weight)

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
	def weight_assignment(self, weights):
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

	def forward(self, states, enemy_states, actions, rnn_hidden_state):
		batch, timesteps, num_agents, _ = states.shape
		_, _, num_enemies, _ = enemy_states.shape
		states = states.reshape(batch*timesteps, num_agents, -1)
		enemy_states = enemy_states.reshape(batch*timesteps, num_enemies, -1)
		actions = actions.reshape(batch*timesteps, num_agents, -1)

		# EMBED STATES KEY & QUERY
		states_embed = self.ally_state_embed_1(states)
		states_query_embed = states_embed.unsqueeze(-2) # Batch size, Num Agents, 1, dim
		# print(states_query_embed.shape)
		# EMBED STATES QUERY
		states_key_embed = states_embed.unsqueeze(1).repeat(1,self.num_agents, 1, 1) # Batch_size, Num agents, Num Agents, dim
		states_key_embed = self.remove_self_loops(states_key_embed) # Batch_size, Num agents, Num Agents - 1, dim
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
			hard_attention_weights = F.gumbel_softmax(self.hard_attention_linear(query_key_concat_intermediate), hard=True, tau=1.0)[:,:,:,1].unsqueeze(-1) # Batch_size, Num agents, Num Agents - 1, 1
			# print(hard_attention_weights.shape)
		else:
			hard_attention_weights = torch.ones(states.shape[0], self.num_agents, self.num_agents-1, 1).to(self.device)
			# print(hard_attention_weights.shape)
		# SOFT ATTENTION
		score = torch.matmul(query_obs,(key_obs).transpose(-2,-1))/math.sqrt(self.d_k) # Batch_size, Num Heads, Num agents, 1, Num Agents - 1
		# print(score.shape)
		weight = F.softmax(score/self.temperature ,dim=-1)*hard_attention_weights.unsqueeze(1).permute(0, 1, 2, 4, 3) # Batch_size, Num Heads, Num agents, 1, Num Agents - 1
		# print(weight.shape)
		weights = self.weight_assignment(weight.squeeze(-2)) # Batch_size, Num Heads, Num agents, Num agents
		# print(weights.shape)

		for head in range(self.num_heads):
			weights[:, head, :, :] = self.attention_dropout(weights[:, head, :, :])

		# EMBED STATE ACTION
		obs_actions = torch.cat([states, actions], dim=-1).to(self.device) # Batch_size, Num agents, dim
		obs_actions_embed_ = self.ally_state_act_embed(obs_actions) #+ self.positional_embedding.unsqueeze(0) # Batch_size, Num agents, dim
		obs_actions_embed = self.remove_self_loops(obs_actions_embed_.unsqueeze(1).repeat(1, self.num_agents, 1, 1)) # Batch_size, Num agents, Num agents - 1, dim
		# print(obs_actions_embed.shape)
		attention_values = torch.stack([self.attention_value[i](obs_actions_embed) for i in range(self.num_heads)], dim=0).permute(1,0,2,3,4) # Batch_size, Num heads, Num agents, Num agents - 1, dim//num_heads
		# print(attention_values.shape)
		aggregated_node_features = torch.matmul(weight, attention_values).squeeze(-2) # Batch_size, Num heads, Num agents, dim//num_heads
		# print(aggregated_node_features.shape)
		aggregated_node_features = aggregated_node_features.permute(0,2,1,3).reshape(states.shape[0], self.num_agents, -1) # Batch_size, Num agents, dim
		# print(aggregated_node_features.shape)
		aggregated_node_features_ = self.attention_value_layer_norm(obs_actions_embed_+aggregated_node_features) # Batch_size, Num agents, dim
		# print(aggregated_node_features_.shape)
		aggregated_node_features = self.attention_value_linear(aggregated_node_features_) # Batch_size, Num agents, dim
		# print(aggregated_node_features.shape)
		aggregated_node_features = self.attention_value_linear_layer_norm(aggregated_node_features_+aggregated_node_features) # Batch_size, Num agents, dim
		# print(aggregated_node_features.shape)
		final_states_embed = self.ally_state_embed_2(states)
		enemy_state_embed = self.enemy_state_embed(enemy_states.reshape(enemy_states.shape[0], -1)).unsqueeze(1).repeat(1, self.num_agents, 1)

		curr_agent_node_features = torch.cat([final_states_embed, enemy_state_embed, aggregated_node_features], dim=-1) # Batch_size, Num agents, dim
		# print(curr_agent_node_features.shape)

		curr_agent_node_features = self.common_layer(curr_agent_node_features) # Batch_size, Num agents, dim
		# print(curr_agent_node_features.shape)
		# curr_agent_node_features = self.RNN(curr_agent_node_features.reshape(-1, curr_agent_node_features.shape[-1]), history.reshape(-1, curr_agent_node_features.shape[-1])).reshape(states.shape[0], self.num_agents, -1) # Batch_size, Num agents, dim
		# print(curr_agent_node_features.shape)
		curr_agent_node_features = curr_agent_node_features.reshape(batch, timesteps, num_agents, -1).permute(0, 2, 1, 3).reshape(batch*num_agents, timesteps, -1)
		output, h = self.RNN(curr_agent_node_features, rnn_hidden_state)
		output = output.reshape(batch, num_agents, timesteps, -1).permute(0, 2, 1, 3).reshape(batch*timesteps, num_agents, -1)
		V_value = self.v_value_layer(output) # Batch_size, Num agents, num_actions
		# print(V_value.shape)

		return V_value.squeeze(-1), weights, score, h
