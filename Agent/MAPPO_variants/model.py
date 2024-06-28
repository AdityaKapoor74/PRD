import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import copy


class PopArt(nn.Module):
	""" Normalize a vector of observations - across the first norm_axes dimensions"""

	def __init__(self, input_shape, num_agents, norm_axes=1, beta=0.99999, per_element_update=False, epsilon=1e-5, device=torch.device("cpu")):
		super(PopArt, self).__init__()

		self.input_shape = input_shape
		self.num_agents = num_agents
		self.norm_axes = norm_axes
		self.epsilon = epsilon
		self.beta = beta
		self.per_element_update = per_element_update
		self.tpdv = dict(dtype=torch.float32, device=device)

		self.running_mean = nn.Parameter(torch.zeros(input_shape), requires_grad=False).to(**self.tpdv)
		self.running_mean_sq = nn.Parameter(torch.zeros(input_shape), requires_grad=False).to(**self.tpdv)
		self.debiasing_term = nn.Parameter(torch.tensor(0.0), requires_grad=False).to(**self.tpdv)

	def reset_parameters(self):
		self.running_mean.zero_()
		self.running_mean_sq.zero_()
		self.debiasing_term.zero_()

	def running_mean_var(self):
		debiased_mean = self.running_mean / self.debiasing_term.clamp(min=self.epsilon)
		debiased_mean_sq = self.running_mean_sq / self.debiasing_term.clamp(min=self.epsilon)
		debiased_var = (debiased_mean_sq - debiased_mean ** 2).clamp(min=1e-2)
		return debiased_mean, debiased_var

	def forward(self, input_vector, mask, train=True):
		# Make sure input is float32
		input_vector_device = input_vector.device
		if type(input_vector) == np.ndarray:
			input_vector = torch.from_numpy(input_vector)
		input_vector = input_vector.to(**self.tpdv)

		if train:
			# Detach input before adding it to running means to avoid backpropping through it on
			# subsequent batches.
			detached_input = input_vector.detach()
			batch_mean = detached_input.sum(dim=tuple(range(self.norm_axes)))/mask.sum(dim=tuple(range(self.norm_axes)))
			batch_sq_mean = (detached_input ** 2).sum(dim=tuple(range(self.norm_axes)))/mask.sum(dim=tuple(range(self.norm_axes)))

			if self.per_element_update:
				batch_size = (mask.reshape(-1, self.num_agents).sum(dim=-1)>0.0).sum()
				weight = self.beta ** batch_size
			else:
				weight = self.beta

			self.running_mean.mul_(weight).add_(batch_mean * (1.0 - weight))
			self.running_mean_sq.mul_(weight).add_(batch_sq_mean * (1.0 - weight))
			self.debiasing_term.mul_(weight).add_(1.0 * (1.0 - weight))

		mean, var = self.running_mean_var()
		out = (input_vector - mean[(None,) * self.norm_axes]) / torch.sqrt(var)[(None,) * self.norm_axes]
		
		return out.to(input_vector_device)

	def denormalize(self, input_vector):
		""" Transform normalized data back into original distribution """
		input_vector_device = input_vector.device
		if type(input_vector) == np.ndarray:
			input_vector = torch.from_numpy(input_vector)
		input_vector = input_vector.to(**self.tpdv)

		mean, var = self.running_mean_var()
		out = input_vector * torch.sqrt(var)[(None,) * self.norm_axes] + mean[(None,) * self.norm_axes]
		
		return out.to(input_vector_device)



def init(module, weight_init, bias_init, gain=1):
	weight_init(module.weight.data, gain=gain)
	if module.bias is not None:
		bias_init(module.bias.data)
	return module

def init_(m, gain=0.01, activate=False):
	if activate:
		gain = nn.init.calculate_gain('relu')
	return init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), gain=gain)



class Policy(nn.Module):
	def __init__(
		self, 
		use_recurrent_policy,
		obs_input_dim, 
		num_actions, 
		num_agents, 
		rnn_num_layers, 
		rnn_hidden_actor,
		device
		):
		super(Policy, self).__init__()

		self.use_recurrent_policy = use_recurrent_policy
		self.rnn_num_layers = rnn_num_layers
		self.rnn_hidden_actor = rnn_hidden_actor

		self.mask_value = torch.tensor(
				torch.finfo(torch.float).min, dtype=torch.float
			)
		self.num_agents = num_agents
		self.num_actions = num_actions
		self.device = device

		self.agent_embedding = nn.Embedding(self.num_agents, self.rnn_hidden_actor)
		self.action_embedding = nn.Embedding(self.num_actions+1, self.rnn_hidden_actor) # we assume the first "last action" to be NON-EXISTENT so one of the embedding represents that

		if self.use_recurrent_policy:
			
			self.obs_embedding = nn.Sequential(
				# nn.LayerNorm(obs_input_dim),
				init_(nn.Linear(obs_input_dim, rnn_hidden_actor), activate=True),
				nn.GELU(),
				)

			self.obs_embed_layer_norm = nn.LayerNorm(self.rnn_hidden_actor)
			
			self.RNN = nn.GRU(input_size=rnn_hidden_actor, hidden_size=rnn_hidden_actor, num_layers=rnn_num_layers, batch_first=True)
			for name, param in self.RNN.named_parameters():
				if 'bias' in name:
					nn.init.constant_(param, 0)
				elif 'weight' in name:
					nn.init.orthogonal_(param)
					
			self.Layer_2 = nn.Sequential(
				nn.LayerNorm(rnn_hidden_actor),
				init_(nn.Linear(rnn_hidden_actor, num_actions), gain=0.01)
				)

			
		else:
			self.obs_embedding = nn.Sequential(
				init_(nn.Linear(obs_input_dim, rnn_hidden_actor), activate=True),
				nn.GELU(),
				)

			self.obs_embed_layer_norm = nn.LayerNorm(self.rnn_hidden_actor)

			self.final_layer = nn.Sequential(
				nn.LayerNorm(rnn_hidden_actor),
				init_(nn.Linear(rnn_hidden_actor, num_actions), gain=0.01)
				)


	def forward(self, local_observations, last_actions, hidden_state, mask_actions):

		batch, timesteps, _, _ = local_observations.shape
		agent_embedding = self.agent_embedding(torch.arange(self.num_agents).to(self.device))[None, None, :, :].expand(batch, timesteps, self.num_agents, self.rnn_hidden_actor)
		last_action_embedding = self.action_embedding(last_actions.long())
		obs_embedding = self.obs_embedding(local_observations)
		final_obs_embedding = self.obs_embed_layer_norm(obs_embedding + last_action_embedding + agent_embedding).permute(0, 2, 1, 3).reshape(batch*self.num_agents, timesteps, -1) # self.obs_embed_layer_norm(obs_embedding + last_action_embedding + agent_embedding).permute(0, 2, 1, 3).reshape(batch*self.num_agents, timesteps, -1)

		if self.use_recurrent_policy:
			hidden_state = hidden_state.reshape(self.rnn_num_layers, batch*self.num_agents, -1)
			output, h = self.RNN(final_obs_embedding, hidden_state)
			output = output.reshape(batch, self.num_agents, timesteps, -1).permute(0, 2, 1, 3)
			logits = self.Layer_2(output)
		else:
			logits = self.final_layer(local_observations)

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


class Global_Q_network(nn.Module):
	def __init__(
		self, 
		use_recurrent_policy,
		ally_obs_input_dim, 
		enemy_obs_input_dim,
		num_heads, 
		num_agents, 
		num_enemies,
		num_teams,
		num_actions, 
		rnn_num_layers,
		comp_emb_shape,
		device, 
		enable_hard_attention, 
		attention_dropout_prob, 
		temperature,
		norm_returns,
		environment,
		):
		super(Global_Q_network, self).__init__()
		
		self.use_recurrent_policy = use_recurrent_policy
		self.num_heads = num_heads
		self.num_agents = num_agents
		self.num_enemies = num_enemies
		self.num_teams = num_teams
		self.num_actions = num_actions
		self.rnn_num_layers = rnn_num_layers
		self.comp_emb_shape = comp_emb_shape
		self.device = device
		self.enable_hard_attention = enable_hard_attention
		self.environment = environment

		self.attention_dropout_prob = attention_dropout_prob

		self.attention_dropout = AttentionDropout(dropout_prob=attention_dropout_prob)

		self.temperature = temperature

		# positional, agent, enemy and team embeddings
		self.agent_embedding = nn.Embedding(self.num_agents, self.comp_emb_shape)
		self.action_embedding = nn.Embedding(self.num_actions, self.comp_emb_shape)
		if "MPE" in self.environment:
			self.team_embedding = nn.Embedding(self.num_teams, self.comp_emb_shape)
		if "StarCraft" in self.environment:
			self.enemy_embedding = nn.Embedding(self.num_enemies, self.comp_emb_shape)
			self.enemy_state_embed = nn.Sequential(
				# nn.LayerNorm(enemy_obs_input_dim),
				init_(nn.Linear(enemy_obs_input_dim, self.comp_emb_shape, bias=True), activate=True),
				nn.GELU(),
				)

		# Embedding Networks
		self.ally_state_embed = nn.Sequential(
			# nn.LayerNorm(ally_obs_input_dim),
			init_(nn.Linear(ally_obs_input_dim, self.comp_emb_shape, bias=True), activate=True),
			nn.GELU(),
			)

		# self.state_embed_layer_norm = nn.LayerNorm(self.comp_emb_shape)

		self.state_action_embed_layer_norm = nn.LayerNorm(self.comp_emb_shape)

		# GLOBAL
		# Key, Query, Attention Value, Hard Attention Networks
		assert 64%self.num_heads == 0
		self.global_key = init_(nn.Linear(self.comp_emb_shape, self.comp_emb_shape), activate=False)
		self.global_query = init_(nn.Linear(self.comp_emb_shape, self.comp_emb_shape), activate=False)
		self.global_attention_value = init_(nn.Linear(self.comp_emb_shape, self.comp_emb_shape), activate=False)

		self.global_attention_value_dropout = nn.Dropout(0.2)
		self.global_attention_value_layer_norm = nn.LayerNorm(self.comp_emb_shape)

		self.global_attention_value_linear = nn.Sequential(
			init_(nn.Linear(self.comp_emb_shape, self.comp_emb_shape), activate=True),
			nn.GELU(),
			init_(nn.Linear(self.comp_emb_shape, self.comp_emb_shape))
			)
		self.global_attention_value_linear_dropout = nn.Dropout(0.2)

		self.global_attention_value_linear_layer_norm = nn.LayerNorm(self.comp_emb_shape)

		# dimesion of key
		self.global_d_k_agents = self.comp_emb_shape

		
		# FCN FINAL LAYER TO GET Q-VALUES
		self.global_common_layer = nn.Sequential(
			init_(nn.Linear(self.comp_emb_shape, self.comp_emb_shape, bias=True), activate=True),
			nn.GELU()
			)

		if self.use_recurrent_policy:
			self.global_RNN = nn.GRU(input_size=self.comp_emb_shape, hidden_size=self.comp_emb_shape, num_layers=self.rnn_num_layers, batch_first=True)
			for name, param in self.global_RNN.named_parameters():
				if 'bias' in name:
					nn.init.constant_(param, 0)
				elif 'weight' in name:
					nn.init.orthogonal_(param)

		
		self.global_q_value_layer = nn.Sequential(
			nn.LayerNorm(self.comp_emb_shape),
			init_(nn.Linear(self.comp_emb_shape, 1, bias=True))
			)



		# INDIV AGENTS
		# Key, Query, Attention Value, Hard Attention Networks
		# assert 64%self.num_heads == 0
		# self.key = init_(nn.Linear(self.comp_emb_shape, self.comp_emb_shape), activate=False)
		# self.query = init_(nn.Linear(self.comp_emb_shape, self.comp_emb_shape), activate=False)
		# self.attention_value = init_(nn.Linear(self.comp_emb_shape, self.comp_emb_shape), activate=False)

		# self.attention_value_dropout = nn.Dropout(0.2)
		# self.attention_value_layer_norm = nn.LayerNorm(self.comp_emb_shape)

		# self.attention_value_linear = nn.Sequential(
		# 	init_(nn.Linear(self.comp_emb_shape, self.comp_emb_shape), activate=True),
		# 	nn.GELU(),
		# 	init_(nn.Linear(self.comp_emb_shape, self.comp_emb_shape))
		# 	)
		# self.attention_value_linear_dropout = nn.Dropout(0.2)

		# self.attention_value_linear_layer_norm = nn.LayerNorm(self.comp_emb_shape)

		# if self.enable_hard_attention:
		# 	self.hard_attention = nn.Sequential(
		# 		init_(nn.Linear(self.comp_emb_shape+self.comp_emb_shape, 2, activate=True))
		# 		)

		# # dimesion of key
		# self.d_k_agents = self.comp_emb_shape

		
		# # FCN FINAL LAYER TO GET Q-VALUES
		# self.common_layer = nn.Sequential(
		# 	init_(nn.Linear(self.comp_emb_shape, self.comp_emb_shape, bias=True), activate=True),
		# 	nn.GELU()
		# 	)

		# self.RNN = nn.GRU(input_size=self.comp_emb_shape, hidden_size=self.comp_emb_shape, num_layers=self.rnn_num_layers, batch_first=True)
		# for name, param in self.RNN.named_parameters():
		# 	if 'bias' in name:
		# 		nn.init.constant_(param, 0)
		# 	elif 'weight' in name:
		# 		nn.init.orthogonal_(param)

		
		# self.q_value_layer = nn.Sequential(
		# 	# nn.LayerNorm(self.comp_emb_shape),
		# 	init_(nn.Linear(self.comp_emb_shape, 1, bias=True))
		# 	)
		

		self.mask_value = torch.tensor(
			torch.finfo(torch.float).min, dtype=torch.float
			)

	
	def get_vector_attention_masks(self, agent_masks):
		attention_masks = copy.deepcopy(1-agent_masks)
		attention_masks[agent_masks[:, :, :] == 0.0] = self.mask_value
		return attention_masks


	def get_attention_masks(self, agent_masks):
		# since we add the attention masks to the score we want to have 0s where the agent is alive and -inf when agent is dead
		attention_masks = copy.deepcopy(1-agent_masks).unsqueeze(-2).repeat(1, 1, self.num_agents, 1)
		# choose columns in each row where the agent is dead and make it -inf
		attention_masks[agent_masks.unsqueeze(-2).repeat(1, 1, self.num_agents, 1)[:, :, :, :] == 0.0] = self.mask_value
		# choose rows of the agent which is dead and make it -inf
		attention_masks[agent_masks.unsqueeze(-2).repeat(1, 1, self.num_agents, 1).transpose(-1,-2)[:, :, :, :] == 0.0] = self.mask_value

		for i in range(self.num_agents):
			attention_masks[:, :, i, i] = self.mask_value
		return attention_masks

	def forward(self, states, enemy_states, actions, global_rnn_hidden_state, agent_masks):
		batch, timesteps, num_agents, _ = states.shape
		states = states.reshape(batch*timesteps, self.num_agents, -1)
		actions = actions.reshape(batch*timesteps, num_agents)

		# extract agent embedding
		agent_embedding = self.agent_embedding(torch.arange(self.num_agents).to(self.device))[None, None, :, :].expand(batch, timesteps, self.num_agents, self.comp_emb_shape).reshape(batch*timesteps, self.num_agents, -1)

		states_embed = self.ally_state_embed(states) + agent_embedding

		if "MPE" in self.environment:
			team_embedding = self.team_embedding(torch.arange(self.num_teams).to(self.device))[None, None, :, None, :].expand(batch, timesteps, self.num_teams, self.num_agents//self.num_teams, self.comp_emb_shape).reshape(batch*timesteps, self.num_agents, self.comp_emb_shape)
			states_embed = states_embed + team_embedding

		if "StarCraft" in self.environment:
			enemy_embedding = self.enemy_embedding(torch.arange(self.num_enemies).to(self.device))[None, None, :, :].expand(batch, timesteps, self.num_enemies, self.comp_emb_shape)
			enemy_state_embed = ((self.enemy_state_embed(enemy_states) + enemy_embedding).sum(dim=2) / self.num_enemies).unsqueeze(2).reshape(batch*timesteps, 1, self.comp_emb_shape)
			states_embed = states_embed + enemy_state_embed

		state_actions_embed = self.state_embed_layer_norm(states_embed + self.action_embedding(actions.long()))

		# KEYS
		key = self.global_key(state_actions_embed).reshape(batch*timesteps, num_agents, self.num_heads, -1).permute(0, 2, 1, 3) # Batch_size, Num Heads, Num agents, dim
		# QUERIES
		# query = self.global_query(self.state_action_embed_layer_norm(state_actions_embed.sum(dim=1)).unsqueeze(1)).reshape(batch*timesteps, self.num_heads, 1, -1) # Batch_size, Num Heads, 1, dim//num_heads
		query = self.global_query((state_actions_embed.sum(dim=1)/self.num_agents).unsqueeze(1)).reshape(batch*timesteps, self.num_heads, 1, -1) # Batch_size, Num Heads, 1, dim//num_heads
		# ATTENTION VALUES
		attention_values = self.global_attention_value(state_actions_embed).reshape(batch*timesteps, num_agents, self.num_heads, -1).permute(0, 2, 1, 3) # Batch_size, Num heads, Num agents, dim//num_heads
		
		# SOFT ATTENTION
		score = torch.matmul(query, (key).transpose(-2,-1)).squeeze(-2)/((self.global_d_k_agents//self.num_heads)**(1/2)) # Batch_size, Num Heads, Num Agents
		
		weights = F.softmax((score/(torch.max(score*(agent_masks.reshape(-1, 1, self.num_agents).repeat(1, self.num_heads, 1)[:, :, :]!=self.mask_value).float(), dim=-1).values-torch.min(score*(agent_masks.reshape(-1, 1, self.num_agents).repeat(1, self.num_heads, 1)[:, :, :]!=self.mask_value).float(), dim=-1).values+1e-5).detach().unsqueeze(-1)) + agent_masks.reshape(-1, 1, self.num_agents).repeat(1, self.num_heads, 1).to(score.device), dim=-1)

		if self.attention_dropout_prob > 0.0:
			for head in range(self.num_heads):
				weights[:, head, :, :] = self.attention_dropout(weights[:, head, :, :])

		aggregated_node_features = (weights.unsqueeze(-1) * attention_values).sum(dim=2) # Batch_size, Num heads, dim//num_heads
		aggregated_node_features = self.global_attention_value_dropout(aggregated_node_features)
		aggregated_node_features = aggregated_node_features.reshape(batch*timesteps, -1) # Batch_size, dim
		aggregated_node_features_ = self.global_attention_value_layer_norm(state_actions_embed.sum(dim=1)+aggregated_node_features) # Batch_size, dim
		aggregated_node_features = self.global_attention_value_linear(aggregated_node_features_) # Batch_size, Num agents, dim
		aggregated_node_features = self.global_attention_value_linear_dropout(aggregated_node_features)
		aggregated_node_features = self.global_attention_value_linear_layer_norm(aggregated_node_features_+aggregated_node_features) # Batch_size, Num agents, dim
		
		multi_agent_system_features = self.global_common_layer(aggregated_node_features) # Batch_size, dim
		
		if self.use_recurrent_policy:
			multi_agent_system_features = multi_agent_system_features.reshape(batch, timesteps, -1)
			rnn_output, h = self.global_RNN(multi_agent_system_features, global_rnn_hidden_state)
			rnn_output = rnn_output.reshape(batch*timesteps, -1)
			joint_Q_value = self.global_q_value_layer(rnn_output) # Batch_size, 1
		else:
			joint_Q_value = self.global_q_value_layer(multi_agent_system_features) # Batch_size, 1

		return joint_Q_value.squeeze(-1), weights, score, h



	# def forward(self, states, enemy_states, actions, rnn_hidden_state, agent_masks):
	# 	batch, timesteps, num_agents, _ = states.shape
	# 	states = states.reshape(batch*timesteps, self.num_agents, -1)
	# 	actions = actions.reshape(batch*timesteps, num_agents)

	# 	# extract agent embedding
	# 	agent_embedding = self.agent_embedding(torch.arange(self.num_agents).to(self.device))[None, None, :, :].expand(batch, timesteps, self.num_agents, self.comp_emb_shape).reshape(batch*timesteps, self.num_agents, -1)

	# 	states_embed = self.ally_state_embed(states) + agent_embedding

	# 	if "MPE" in self.environment:
	# 		team_embedding = self.team_embedding(torch.arange(self.num_teams).to(self.device))[None, None, :, None, :].expand(batch, timesteps, self.num_teams, self.num_agents//self.num_teams, self.comp_emb_shape).reshape(batch*timesteps, self.num_agents, self.comp_emb_shape)
	# 		states_embed = states_embed + team_embedding

	# 	if "StarCraft" in self.environment:
	# 		enemy_embedding = self.enemy_embedding(torch.arange(self.num_enemies).to(self.device))[None, None, :, :].expand(batch, timesteps, self.num_enemies, self.comp_emb_shape)
	# 		enemy_state_embed = (self.enemy_state_embed(enemy_states) + enemy_embedding).sum(dim=2).unsqueeze(2).reshape(batch*timesteps, 1, self.comp_emb_shape)
	# 		states_embed = states_embed + enemy_state_embed

	# 	states_embed = self.state_embed_layer_norm(states_embed)

	# 	# KEYS
	# 	key_obs = self.key(states_embed).reshape(batch*timesteps, num_agents, self.num_heads, -1).permute(0, 2, 1, 3) # Batch_size, Num Heads, Num agents, dim
	# 	# QUERIES
	# 	query_obs = self.query(states_embed).reshape(batch*timesteps, num_agents, self.num_heads, -1).permute(0, 2, 1, 3) # Batch_size, Num Heads, Num agents, dim

	# 	# HARD ATTENTION
	# 	if self.enable_hard_attention:
	# 		query_key_concat = torch.cat([query_obs.unsqueeze(3).repeat(1,1,1,self.num_agents,1), key_obs.unsqueeze(2).repeat(1,1,self.num_agents,1,1)], dim=-1) # Batch_size, Num Heads, Num agents, Num Agents, dim
	# 		query_key_concat_intermediate = self.hard_attention(query_key_concat) # Batch_size, Num Heads, Num agents, Num agents-1, dim
	# 		hard_attention_weights = F.gumbel_softmax(query_key_concat_intermediate, hard=True, tau=1.0)[:,:,:,:,1] # Batch_size, Num Heads, Num agents, Num Agents, 1			
	# 		for i in range(self.num_agents):
	# 			hard_attention_weights[:,:,i,i] = 1.0
	# 	else:
	# 		hard_attention_weights = torch.ones(states.shape[0], self.num_heads, self.num_agents, self.num_agents).float().to(self.device)

	# 	# SOFT ATTENTION
	# 	score = torch.matmul(query_obs,(key_obs).transpose(-2,-1))/((self.d_k_agents//self.num_heads)**(1/2)) # Batch_size, Num Heads, Num agents, Num Agents
	# 	attention_masks = self.get_attention_masks(agent_masks).reshape(batch*timesteps, num_agents, num_agents).unsqueeze(1).repeat(1, self.num_heads, 1, 1)
		
	# 	weights = F.softmax((score/(torch.max(score*(attention_masks[:, :, :, :]!=self.mask_value).float(), dim=-1).values-torch.min(score*(attention_masks[:, :, :, :]!=self.mask_value).float(), dim=-1).values+1e-5).detach().unsqueeze(-1)) + attention_masks.reshape(*score.shape).to(score.device), dim=-1) #* hard_attention_weights.unsqueeze(1).permute(0, 1, 2, 4, 3) # Batch_size, Num Heads, Num agents, 1, Num Agents - 1

	# 	if self.attention_dropout_prob > 0.0:
	# 		for head in range(self.num_heads):
	# 			weights[:, head, :, :] = self.attention_dropout(weights[:, head, :, :])

	# 	# weights = weights.clone()

	# 	prd_weights = F.softmax((score/(torch.max(score*(attention_masks[:, :, :, :]!=self.mask_value).float(), dim=-2).values-torch.min(score*(attention_masks[:, :, :, :]!=self.mask_value).float(), dim=-2).values+1e-5).detach().unsqueeze(-1)) + attention_masks.reshape(*score.shape).to(score.device), dim=-2)
		
	# 	for i in range(self.num_agents):
	# 		weights[:, :, i, i] = 1.0 # since weights[:, :, i, i] = 0.0
	# 		prd_weights[:, :, i, i] = 1.0

	# 	weights = weights * agent_masks.reshape(batch*timesteps, 1, self.num_agents, 1).repeat(1, self.num_heads, 1, self.num_agents)
	# 	weights = weights * agent_masks.reshape(batch*timesteps, 1, 1, self.num_agents).repeat(1, self.num_heads, self.num_agents, 1)
	# 	prd_weights = prd_weights * agent_masks.reshape(batch*timesteps, 1, self.num_agents, 1).repeat(1, self.num_heads, 1, self.num_agents)
	# 	prd_weights = prd_weights * agent_masks.reshape(batch*timesteps, 1, 1, self.num_agents).repeat(1, self.num_heads, self.num_agents, 1)


	# 	obs_actions_embed = states_embed + self.action_embedding(actions.long())

	# 	attention_values = self.attention_value(obs_actions_embed).reshape(batch*timesteps, num_agents, self.num_heads, -1).permute(0, 2, 1, 3) #torch.stack([self.attention_value[i](obs_actions_embed) for i in range(self.num_heads)], dim=0).permute(1,0,2,3,4) # Batch_size, Num heads, Num agents, Num agents - 1, dim//num_heads
		
	# 	global_score = score.sum(dim=-2)*agent_masks.reshape(batch*timesteps, 1, self.num_agents).to(score.device)/(agent_masks.reshape(-1, self.num_agents).sum(dim=-1)+1e-5).reshape(-1, 1, 1)
		
	# 	global_weights = weights.sum(dim=-2)/(agent_masks.reshape(-1, self.num_agents).sum(dim=-1)+1e-5).reshape(-1, 1, 1)
	# 	global_weights_updated = global_weights / (global_weights.sum(dim=-1, keepdim=True).detach()+1e-5)

	# 	aggregated_node_features = (global_weights_updated.unsqueeze(-1)*attention_values).sum(dim=-2) # Batch_size, Num heads, dim//num_heads
	# 	aggregated_node_features = self.attention_value_dropout(aggregated_node_features)
	# 	aggregated_node_features = aggregated_node_features.reshape(states.shape[0], -1) # Batch_size, dim
	# 	aggregated_node_features_ = self.attention_value_layer_norm(obs_actions_embed.sum(dim=-2)+aggregated_node_features) # Batch_size, dim
	# 	aggregated_node_features = self.attention_value_linear(aggregated_node_features_) # Batch_size, dim
	# 	aggregated_node_features = self.attention_value_linear_dropout(aggregated_node_features)
	# 	aggregated_node_features = self.attention_value_linear_layer_norm(aggregated_node_features_+aggregated_node_features) # Batch_size, Num agents, dim

	# 	curr_agent_node_features = self.common_layer(aggregated_node_features) # Batch_size, dim
		
	# 	curr_agent_node_features = curr_agent_node_features.reshape(batch, timesteps, -1)
	# 	rnn_output, global_h = self.RNN(curr_agent_node_features, rnn_hidden_state)
	# 	rnn_output = rnn_output.reshape(batch*timesteps, -1)
	# 	global_Q_value = self.q_value_layer(rnn_output) # Batch_size, 1
		
		
	# 	return global_Q_value, global_weights_updated, prd_weights, global_score, score, global_h



class Q_network(nn.Module):
	def __init__(
		self, 
		use_recurrent_policy,
		ally_obs_input_dim, 
		enemy_obs_input_dim,
		num_heads, 
		num_agents, 
		num_enemies,
		num_teams,
		num_actions, 
		rnn_num_layers,
		comp_emb_shape,
		device, 
		enable_hard_attention, 
		attention_dropout_prob, 
		temperature,
		norm_returns,
		environment,
		):
		super(Q_network, self).__init__()
		
		self.use_recurrent_policy = use_recurrent_policy,
		self.num_heads = num_heads
		self.num_agents = num_agents
		self.num_enemies = num_enemies
		self.num_teams = num_teams
		self.num_actions = num_actions
		self.rnn_num_layers = rnn_num_layers
		self.comp_emb_shape = comp_emb_shape
		self.device = device
		self.enable_hard_attention = enable_hard_attention
		self.environment = environment
		self.attention_dropout_prob = attention_dropout_prob

		self.attention_dropout = AttentionDropout(dropout_prob=attention_dropout_prob)

		self.temperature = temperature

		# positional, agent, enemy and team embeddings
		self.agent_embedding = nn.Embedding(self.num_agents, self.comp_emb_shape)
		self.action_embedding = nn.Embedding(self.num_actions, self.comp_emb_shape)
		if "MPE" in self.environment:
			self.team_embedding = nn.Embedding(self.num_teams, self.comp_emb_shape)
		if "StarCraft" in self.environment:
			self.enemy_embedding = nn.Embedding(self.num_enemies, self.comp_emb_shape)
			self.enemy_state_embed = nn.Sequential(
				# nn.LayerNorm(enemy_obs_input_dim),
				init_(nn.Linear(enemy_obs_input_dim, self.comp_emb_shape, bias=True), activate=True),
				nn.GELU(),
				)

		# Embedding Networks
		self.ally_state_embed = nn.Sequential(
			# nn.LayerNorm(ally_obs_input_dim),
			init_(nn.Linear(ally_obs_input_dim, self.comp_emb_shape, bias=True), activate=True),
			nn.GELU(),
			)

		self.state_embed_layer_norm = nn.LayerNorm(self.comp_emb_shape)
			
		# Key, Query, Attention Value, Hard Attention Networks
		assert 64%self.num_heads == 0
		self.key = init_(nn.Linear(self.comp_emb_shape, self.comp_emb_shape), activate=False)
		self.query = init_(nn.Linear(self.comp_emb_shape, self.comp_emb_shape), activate=False)
		self.attention_value = init_(nn.Linear(self.comp_emb_shape, self.comp_emb_shape), activate=False)

		self.attention_value_dropout = nn.Dropout(0.2)
		self.attention_value_layer_norm = nn.LayerNorm(64)

		self.attention_value_linear = nn.Sequential(
			init_(nn.Linear(self.comp_emb_shape, self.comp_emb_shape, bias=True), activate=True),
			nn.GELU(),
			init_(nn.Linear(self.comp_emb_shape, self.comp_emb_shape, bias=True)),
			)
		self.attention_value_linear_dropout = nn.Dropout(0.2)

		self.attention_value_linear_layer_norm = nn.LayerNorm(self.comp_emb_shape)

		if self.enable_hard_attention:
			self.hard_attention = nn.Sequential(
				init_(nn.Linear(self.comp_emb_shape+self.comp_emb_shape, 2, bias=True))
				)

		# dimesion of key
		self.d_k_agents = self.comp_emb_shape

		self.common_layer = nn.Sequential(
			init_(nn.Linear(self.comp_emb_shape, self.comp_emb_shape, bias=True), activate=True),
			nn.GELU()
			)

		if self.use_recurrent_policy:
			self.RNN = nn.GRU(input_size=self.comp_emb_shape, hidden_size=self.comp_emb_shape, num_layers=self.rnn_num_layers, batch_first=True)
			for name, param in self.RNN.named_parameters():
				if 'bias' in name:
					nn.init.constant_(param, 0)
				elif 'weight' in name:
					nn.init.orthogonal_(param)

		

		self.q_value_layer = nn.Sequential(
			nn.LayerNorm(self.comp_emb_shape),
			init_(nn.Linear(self.comp_emb_shape, 1, bias=True))
			)

		self.mask_value = torch.tensor(
			torch.finfo(torch.float).min, dtype=torch.float
			)


	def get_attention_masks(self, agent_masks):
		# since we add the attention masks to the score we want to have 0s where the agent is alive and -inf when agent is dead
		attention_masks = copy.deepcopy(1-agent_masks).unsqueeze(-2).repeat(1, 1, self.num_agents, 1)
		# choose columns in each row where the agent is dead and make it -inf
		attention_masks[agent_masks.unsqueeze(-2).repeat(1, 1, self.num_agents, 1)[:, :, :, :] == 0.0] = self.mask_value
		# choose rows of the agent which is dead and make it -inf
		attention_masks[agent_masks.unsqueeze(-2).repeat(1, 1, self.num_agents, 1).transpose(-1,-2)[:, :, :, :] == 0.0] = self.mask_value

		for i in range(self.num_agents):
			attention_masks[:, :, i, i] = self.mask_value
		return attention_masks


	def forward(self, states, enemy_states, actions, rnn_hidden_state, agent_masks):
		batch, timesteps, num_agents, _ = states.shape
		states = states.reshape(batch*timesteps, self.num_agents, -1)
		actions = actions.reshape(batch*timesteps, num_agents)

		# extract agent embedding
		agent_embedding = self.agent_embedding(torch.arange(self.num_agents).to(self.device))[None, None, :, :].expand(batch, timesteps, self.num_agents, self.comp_emb_shape).reshape(batch*timesteps, self.num_agents, -1)

		states_embed = self.ally_state_embed(states) + agent_embedding

		if "MPE" in self.environment:
			team_embedding = self.team_embedding(torch.arange(self.num_teams).to(self.device))[None, None, :, None, :].expand(batch, timesteps, self.num_teams, self.num_agents//self.num_teams, self.comp_emb_shape).reshape(batch*timesteps, self.num_agents, self.comp_emb_shape)
			states_embed = states_embed + team_embedding

		if "StarCraft" in self.environment:
			enemy_embedding = self.enemy_embedding(torch.arange(self.num_enemies).to(self.device))[None, None, :, :].expand(batch, timesteps, self.num_enemies, self.comp_emb_shape)
			enemy_state_embed = (self.enemy_state_embed(enemy_states) + enemy_embedding).sum(dim=2).unsqueeze(2).reshape(batch*timesteps, 1, self.comp_emb_shape)
			states_embed = states_embed + enemy_state_embed
		
		states_embed = self.state_embed_layer_norm(states_embed)

		# KEYS
		key_obs = self.key(states_embed).reshape(batch*timesteps, num_agents, self.num_heads, -1).permute(0, 2, 1, 3) # Batch_size, Num Heads, Num agents, dim
		# QUERIES
		query_obs = self.query(states_embed).reshape(batch*timesteps, num_agents, self.num_heads, -1).permute(0, 2, 1, 3) # Batch_size, Num Heads, Num agents, dim

		# HARD ATTENTION
		if self.enable_hard_attention:
			query_key_concat = torch.cat([query_obs.unsqueeze(3).repeat(1,1,1,self.num_agents,1), key_obs.unsqueeze(2).repeat(1,1,self.num_agents,1,1)], dim=-1) # Batch_size, Num Heads, Num agents, Num Agents, dim
			query_key_concat_intermediate = self.hard_attention(query_key_concat) # Batch_size, Num Heads, Num agents, Num agents-1, dim
			hard_attention_weights = F.gumbel_softmax(query_key_concat_intermediate, hard=True, tau=1.0)[:,:,:,:,1] # Batch_size, Num Heads, Num agents, Num Agents, 1			
			for i in range(self.num_agents):
				hard_attention_weights[:,:,i,i] = 1.0
		else:
			hard_attention_weights = torch.ones(states.shape[0], self.num_heads, self.num_agents, self.num_agents).float().to(self.device)
			

		# SOFT ATTENTION
		score = torch.matmul(query_obs,(key_obs).transpose(-2,-1))/(self.d_k_agents//self.num_heads)**(1/2) # Batch_size, Num Heads, Num agents, Num Agents
		
		attention_masks = self.get_attention_masks(agent_masks).reshape(batch*timesteps, num_agents, num_agents).unsqueeze(1).repeat(1, self.num_heads, 1, 1)
		attention_masks = attention_masks + (1-hard_attention_weights)*self.mask_value
		weights = F.softmax((score/(torch.max(score*(attention_masks[:, :, :, :]!=self.mask_value).float(), dim=-1).values-torch.min(score*(attention_masks[:, :, :, :]!=self.mask_value).float(), dim=-1).values+1e-5).detach().unsqueeze(-1)) + attention_masks.reshape(*score.shape).to(score.device), dim=-1) # Batch_size, Num Heads, Num agents, Num Agents
		
		if self.attention_dropout_prob > 0.0:
			for head in range(self.num_heads):
				weights[:, head, :, :] = self.attention_dropout(weights[:, head, :, :])

		final_weights = weights.clone()
		prd_weights = F.softmax((score/(torch.max(score*(attention_masks[:, :, :, :]!=self.mask_value).float(), dim=-2).values-torch.min(score*(attention_masks[:, :, :, :]!=self.mask_value).float(), dim=-2).values+1e-5).detach().unsqueeze(-1)) + attention_masks.reshape(*score.shape).to(score.device), dim=-2) # Batch_size, Num Heads, Num agents, Num Agents
		for i in range(self.num_agents):
			final_weights[:, :, i, i] = 1.0 # since weights[:, :, i, i] = 0.0
			prd_weights[:, :, i, i] = 1.0

		final_weights = final_weights * agent_masks.reshape(batch*timesteps, 1, self.num_agents, 1).repeat(1, self.num_heads, 1, self.num_agents)
		final_weights = final_weights * agent_masks.reshape(batch*timesteps, 1, 1, self.num_agents).repeat(1, self.num_heads, self.num_agents, 1)
		prd_weights = prd_weights * agent_masks.reshape(batch*timesteps, 1, self.num_agents, 1).repeat(1, self.num_heads, 1, self.num_agents)
		prd_weights = prd_weights * agent_masks.reshape(batch*timesteps, 1, 1, self.num_agents).repeat(1, self.num_heads, self.num_agents, 1)

		
		# EMBED STATE ACTION
		obs_actions_embed = states_embed + self.action_embedding(actions.long())

		attention_values = self.attention_value(obs_actions_embed).reshape(batch*timesteps, num_agents, self.num_heads, -1).permute(0, 2, 1, 3) #torch.stack([self.attention_value[i](obs_actions_embed) for i in range(self.num_heads)], dim=0).permute(1,0,2,3,4) # Batch_size, Num heads, Num agents, Num agents - 1, dim//num_heads
		
		aggregated_node_features = torch.matmul(final_weights, attention_values) # Batch_size, Num heads, Num agents, dim//num_heads
		aggregated_node_features = self.attention_value_dropout(aggregated_node_features)
		aggregated_node_features = aggregated_node_features.permute(0,2,1,3).reshape(batch*timesteps, self.num_agents, -1) # Batch_size, Num agents, dim
		aggregated_node_features_ = self.attention_value_layer_norm(obs_actions_embed+aggregated_node_features) # Batch_size, Num agents, dim
		aggregated_node_features = self.attention_value_linear(aggregated_node_features_) # Batch_size, Num agents, dim
		aggregated_node_features = self.attention_value_linear_dropout(aggregated_node_features)
		aggregated_node_features = self.attention_value_linear_layer_norm(aggregated_node_features_+aggregated_node_features) # Batch_size, Num agents, dim
		
		curr_agent_node_features = self.common_layer(aggregated_node_features) # Batch_size, Num agents, dim
		
		if self.use_recurrent_policy:
			curr_agent_node_features = curr_agent_node_features.reshape(batch, timesteps, num_agents, -1).permute(0, 2, 1, 3).reshape(batch*num_agents, timesteps, -1)
			rnn_output, h = self.RNN(curr_agent_node_features, rnn_hidden_state)
			rnn_output = rnn_output.reshape(batch, num_agents, timesteps, -1).permute(0, 2, 1, 3).reshape(batch*timesteps, num_agents, -1)
			Q_value = self.q_value_layer(rnn_output) # Batch_size, Num agents, 1
		else:
			Q_value = self.q_value_layer(curr_agent_node_features) # Batch_size, Num agents, 1

		return Q_value.squeeze(-1), prd_weights, score, h


class V_network(nn.Module):
	def __init__(
		self, 
		use_recurrent_policy,
		ally_obs_input_dim, 
		enemy_obs_input_dim,
		num_heads, 
		num_agents, 
		num_enemies,
		num_teams,
		num_actions,
		rnn_num_layers,
		comp_emb_shape,
		device, 
		enable_hard_attention, 
		attention_dropout_prob, 
		temperature,
		norm_returns,
		environment,
		experiment_type,
		):
		super(V_network, self).__init__()
		
		self.use_recurrent_policy = use_recurrent_policy,
		self.num_heads = num_heads
		self.num_agents = num_agents
		self.num_enemies = num_enemies
		self.num_teams = num_teams
		self.num_actions = num_actions
		self.rnn_num_layers = rnn_num_layers
		self.comp_emb_shape = comp_emb_shape
		self.device = device
		self.enable_hard_attention = enable_hard_attention
		self.environment = environment
		self.experiment_type = experiment_type
		self.attention_dropout_prob = attention_dropout_prob

		self.attention_dropout = AttentionDropout(dropout_prob=attention_dropout_prob)

		self.temperature = temperature

		# positional, agent, enemy and team embeddings
		self.agent_embedding = nn.Embedding(self.num_agents, self.comp_emb_shape)
		self.action_embedding = nn.Embedding(self.num_actions, self.comp_emb_shape)
		if "MPE" in self.environment:
			self.team_embedding = nn.Embedding(self.num_teams, self.comp_emb_shape)
		if "StarCraft" in self.environment:
			self.enemy_embedding = nn.Embedding(self.num_enemies, self.comp_emb_shape)
			self.enemy_state_embed = nn.Sequential(
				# nn.LayerNorm(enemy_obs_input_dim),
				init_(nn.Linear(enemy_obs_input_dim, self.comp_emb_shape, bias=True), activate=True),
				nn.GELU(),
				)

		# Embedding Networks
		self.ally_state_embed = nn.Sequential(
			# nn.LayerNorm(ally_obs_input_dim),
			init_(nn.Linear(ally_obs_input_dim, self.comp_emb_shape, bias=True), activate=True),
			nn.GELU(),
			)

		self.state_embed_layer_norm = nn.LayerNorm(self.comp_emb_shape)

		# Key, Query, Attention Value, Hard Attention Networks
		assert 64%self.num_heads == 0
		self.key = init_(nn.Linear(self.comp_emb_shape, self.comp_emb_shape), activate=False)
		self.query = init_(nn.Linear(self.comp_emb_shape, self.comp_emb_shape), activate=False)
		self.attention_value = init_(nn.Linear(self.comp_emb_shape, self.comp_emb_shape), activate=False)

		self.attention_value_dropout = nn.Dropout(0.2)
		self.attention_value_layer_norm = nn.LayerNorm(self.comp_emb_shape)

		self.attention_value_linear = nn.Sequential(
			init_(nn.Linear(self.comp_emb_shape, self.comp_emb_shape, bias=True), activate=True),
			nn.GELU(),
			init_(nn.Linear(self.comp_emb_shape, self.comp_emb_shape, bias=True))
			)
		self.attention_value_linear_dropout = nn.Dropout(0.2)

		self.attention_value_linear_layer_norm = nn.LayerNorm(self.comp_emb_shape)

		if self.enable_hard_attention:
			self.hard_attention = nn.Sequential(
				init_(nn.Linear(self.comp_emb_shape+self.comp_emb_shape, 2, bias=True))
				)

		# dimesion of key
		self.d_k_agents = self.comp_emb_shape


		self.common_layer = nn.Sequential(
			init_(nn.Linear(self.comp_emb_shape, self.comp_emb_shape, bias=True), activate=True),
			nn.GELU()
			)

		if self.use_recurrent_policy:
			self.RNN = nn.GRU(input_size=64, hidden_size=64, num_layers=self.rnn_num_layers, batch_first=True)
			for name, param in self.RNN.named_parameters():
				if 'bias' in name:
					nn.init.constant_(param, 0)
				elif 'weight' in name:
					nn.init.orthogonal_(param)

		

		self.v_value_layer = nn.Sequential(
			nn.LayerNorm(self.comp_emb_shape),
			init_(nn.Linear(self.comp_emb_shape, 1, bias=True))
			)

		self.mask_value = torch.tensor(
			torch.finfo(torch.float).min, dtype=torch.float
			)


	def get_attention_masks(self, agent_masks):
		# since we add the attention masks to the score we want to have 0s where the agent is alive and -inf when agent is dead
		attention_masks = copy.deepcopy(1-agent_masks).unsqueeze(-2).repeat(1, 1, self.num_agents, 1)
		# choose columns in each row where the agent is dead and make it -inf
		attention_masks[agent_masks.unsqueeze(-2).repeat(1, 1, self.num_agents, 1)[:, :, :, :] == 0.0] = self.mask_value
		# choose rows of the agent which is dead and make it -inf
		attention_masks[agent_masks.unsqueeze(-2).repeat(1, 1, self.num_agents, 1).transpose(-1,-2)[:, :, :, :] == 0.0] = self.mask_value

		for i in range(self.num_agents):
			attention_masks[:, :, i, i] = self.mask_value
		return attention_masks


	def forward(self, states, enemy_states, actions, rnn_hidden_state, agent_masks):
		batch, timesteps, num_agents, _ = states.shape
		states = states.reshape(batch*timesteps, num_agents, -1)
		actions = actions.reshape(batch*timesteps, num_agents)

		# extract agent embedding
		agent_embedding = self.agent_embedding(torch.arange(self.num_agents).to(self.device))[None, None, :, :].expand(batch, timesteps, self.num_agents, self.comp_emb_shape).reshape(batch*timesteps, self.num_agents, -1)
		

		# EMBED STATES KEY & QUERY
		states_embed = self.ally_state_embed(states) + agent_embedding

		if "MPE" in self.environment:
			team_embedding = self.team_embedding(torch.arange(self.num_teams).to(self.device))[None, None, :, None, :].expand(batch, timesteps, self.num_teams, self.num_agents//self.num_teams, self.comp_emb_shape).reshape(batch*timesteps, self.num_agents, self.comp_emb_shape)
			states_embed = states_embed + team_embedding

		if "StarCraft" in self.environment:
			enemy_embedding = self.enemy_embedding(torch.arange(self.num_enemies).to(self.device))[None, None, :, :].expand(batch, timesteps, self.num_enemies, self.comp_emb_shape)
			enemy_state_embed = (self.enemy_state_embed(enemy_states) + enemy_embedding).sum(dim=2).unsqueeze(2).reshape(batch*timesteps, 1, self.comp_emb_shape)
			states_embed = states_embed + enemy_state_embed
		
		states_embed = self.state_embed_layer_norm(states_embed)

		# KEYS
		key_obs = self.key(states_embed).reshape(batch*timesteps, num_agents, self.num_heads, -1).permute(0, 2, 1, 3) # Batch_size, Num Heads, Num agents, dim
		# QUERIES
		query_obs = self.query(states_embed).reshape(batch*timesteps, num_agents, self.num_heads, -1).permute(0, 2, 1, 3) # Batch_size, Num Heads, Num agents, dim

		# HARD ATTENTION
		if self.enable_hard_attention:
			query_key_concat = torch.cat([query_obs.unsqueeze(3).repeat(1,1,1,self.num_agents,1), key_obs.unsqueeze(2).repeat(1,1,self.num_agents,1,1)], dim=-1) # Batch_size, Num Heads, Num agents, Num Agents, dim
			query_key_concat_intermediate = self.hard_attention(query_key_concat) # Batch_size, Num Heads, Num agents, Num agents-1, dim
			hard_attention_weights = F.gumbel_softmax(query_key_concat_intermediate, hard=True, tau=1.0)[:,:,:,:,1] # Batch_size, Num Heads, Num agents, Num Agents - 1, 1			
			for i in range(self.num_agents):
				hard_attention_weights[:,:,i,i] = 1.0
		else:
			hard_attention_weights = torch.ones(states.shape[0], self.num_heads, self.num_agents, self.num_agents).float().to(self.device)
			
		# SOFT ATTENTION
		score = torch.matmul(query_obs,(key_obs).transpose(-2,-1))/(self.d_k_agents//self.num_heads)**(1/2) # Batch_size, Num Heads, Num agents, Num Agents
		
		attention_masks = self.get_attention_masks(agent_masks).reshape(batch*timesteps, num_agents, num_agents).unsqueeze(1).repeat(1, self.num_heads, 1, 1)
		attention_masks = attention_masks + (1-hard_attention_weights)*self.mask_value
		
		weights = F.softmax((score/(torch.max(score*(attention_masks[:, :, :, :]!=self.mask_value).float(), dim=-1).values-torch.min(score*(attention_masks[:, :, :, :]!=self.mask_value).float(), dim=-1).values+1e-5).detach().unsqueeze(-1)) + attention_masks.reshape(*score.shape).to(score.device), dim=-1) # Batch_size, Num Heads, Num agents, Num Agents

		if self.attention_dropout_prob > 0.0:
			for head in range(self.num_heads):
				weights[:, head, :, :] = self.attention_dropout(weights[:, head, :, :])

		final_weights = weights.clone()
		for i in range(self.num_agents):
			final_weights[:, :, i, i] = 1.0

		final_weights = final_weights * agent_masks.reshape(batch*timesteps, 1, self.num_agents, 1).repeat(1, self.num_heads, 1, self.num_agents)
		final_weights = final_weights * agent_masks.reshape(batch*timesteps, 1, 1, self.num_agents).repeat(1, self.num_heads, self.num_agents, 1)

		# EMBED STATE ACTION
		# need to get rid of actions of current agent in question for calculating the baseline
		prd_masks = torch.ones(batch*timesteps, self.num_agents, self.num_agents).to(self.device)
		for i in range(self.num_agents):
			prd_masks[:, i, i] = 0.0

		obs_actions_embed = states_embed.unsqueeze(1).repeat(1, self.num_agents, 1, 1).to(self.device) + self.action_embedding(actions.long()).unsqueeze(1).repeat(1, self.num_agents, 1, 1).to(self.device) * prd_masks.unsqueeze(-1) # Batch_size, Num agents, Num agents, dim
		

		attention_values = self.attention_value(obs_actions_embed).reshape(batch*timesteps, num_agents, num_agents, self.num_heads, -1).permute(0, 3, 1, 2, 4) # Batch_size, Num heads, Num agents, Num agents, dim//num_heads
		aggregated_node_features = (final_weights.unsqueeze(-1) * attention_values).sum(dim=-2)
		obs_actions_embed_ = obs_actions_embed.sum(dim=-2) # summing so that we can use it later in layer norm for aggregation
		
		
		aggregated_node_features = self.attention_value_dropout(aggregated_node_features)
		aggregated_node_features = aggregated_node_features.permute(0,2,1,3).reshape(states.shape[0], self.num_agents, -1) # Batch_size, Num agents, dim
		aggregated_node_features_ = self.attention_value_layer_norm(obs_actions_embed_+aggregated_node_features) # Batch_size, Num agents, dim
		aggregated_node_features = self.attention_value_linear(aggregated_node_features_) # Batch_size, Num agents, dim
		aggregated_node_features = self.attention_value_linear_dropout(aggregated_node_features)
		aggregated_node_features = self.attention_value_linear_layer_norm(aggregated_node_features_+aggregated_node_features) # Batch_size, Num agents, dim
		
		curr_agent_node_features = torch.cat([aggregated_node_features], dim=-1) # Batch_size, Num agents, dim
		curr_agent_node_features = self.common_layer(curr_agent_node_features) # Batch_size, Num agents, dim
		
		if self.use_recurrent_policy:
			curr_agent_node_features = curr_agent_node_features.reshape(batch, timesteps, num_agents, -1).permute(0, 2, 1, 3).reshape(batch*num_agents, timesteps, -1)
			rnn_output, h = self.RNN(curr_agent_node_features, rnn_hidden_state)
			rnn_output = rnn_output.reshape(batch, num_agents, timesteps, -1).permute(0, 2, 1, 3).reshape(batch*timesteps, num_agents, -1)
			V_value = self.v_value_layer(rnn_output) # Batch_size, Num agents, 1
		else:
			V_value = self.v_value_layer(curr_agent_node_features) # Batch_size, Num agents, 1

		return V_value.squeeze(-1), final_weights, score, h
