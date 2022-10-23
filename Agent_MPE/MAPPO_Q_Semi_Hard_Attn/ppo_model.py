from typing import Any, List, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import datetime
import math

class RolloutBuffer:
	def __init__(self):
		self.states_actor = []
		self.probs = []
		self.logprobs = []
		self.actions = []
		self.one_hot_actions = []


		self.rewards = []
		self.dones = []

		
		self.states_critic = []
	

	def clear(self):
		del self.actions[:]
		del self.states_critic[:]
		del self.states_actor[:]
		del self.probs[:]
		del self.one_hot_actions[:]
		del self.logprobs[:]
		del self.rewards[:]
		del self.dones[:]


class PopArt(torch.nn.Module):
	
	def __init__(self, input_shape, output_shape, norm_axes=1, beta=0.99999, epsilon=1e-5, device=torch.device("cpu")):
		
		super(PopArt, self).__init__()

		self.beta = beta
		self.epsilon = epsilon
		self.norm_axes = norm_axes
		self.device=device

		self.input_shape = input_shape
		self.output_shape = output_shape

		self.weight = nn.Parameter(torch.Tensor(output_shape, input_shape)).to(self.device)
		self.bias = nn.Parameter(torch.Tensor(output_shape)).to(self.device)
		
		self.stddev = nn.Parameter(torch.ones(output_shape, dtype=torch.float64), requires_grad=False).to(self.device)
		self.mean = nn.Parameter(torch.zeros(output_shape, dtype=torch.float64), requires_grad=False).to(self.device)
		self.mean_sq = nn.Parameter(torch.zeros(output_shape, dtype=torch.float64), requires_grad=False).to(self.device)
		self.debiasing_term = nn.Parameter(torch.tensor(0.0), requires_grad=False).to(self.device)

		self.reset_parameters()

	def reset_parameters(self):
		torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
		if self.bias is not None:
			fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.weight)
			bound = 1 / math.sqrt(fan_in)
			torch.nn.init.uniform_(self.bias, -bound, bound)
		self.mean.zero_()
		self.mean_sq.zero_()
		self.debiasing_term.zero_()

	def forward(self, input_vector):
		if type(input_vector) == np.ndarray:
			input_vector = torch.from_numpy(input_vector)

		return F.linear(input_vector.float(), self.weight.float(), self.bias.float())
	
	@torch.no_grad()
	def update(self, input_vector):
		if type(input_vector) == np.ndarray:
			input_vector = torch.from_numpy(input_vector)
		
		old_mean, old_var = self.debiased_mean_var()
		old_stddev = torch.sqrt(old_var)

		batch_mean = input_vector.mean(dim=tuple(range(self.norm_axes)))
		batch_sq_mean = (input_vector ** 2).mean(dim=tuple(range(self.norm_axes)))


		self.mean.mul_(self.beta).add_(batch_mean * (1.0 - self.beta))
		self.mean_sq.mul_(self.beta).add_(batch_sq_mean * (1.0 - self.beta))
		self.debiasing_term.mul_(self.beta).add_(1.0 * (1.0 - self.beta))

		self.stddev = (self.mean_sq - self.mean ** 2).sqrt().clamp(min=1e-4)
		
		new_mean, new_var = self.debiased_mean_var()
		new_stddev = torch.sqrt(new_var)

		self.weight = self.weight * old_stddev.unsqueeze(-1) / new_stddev.unsqueeze(-1)
		self.bias = (old_stddev * self.bias + old_mean - new_mean) / new_stddev

	def debiased_mean_var(self):
		debiased_mean = self.mean / self.debiasing_term.clamp(min=self.epsilon)
		debiased_mean_sq = self.mean_sq / self.debiasing_term.clamp(min=self.epsilon)
		debiased_var = (debiased_mean_sq - debiased_mean ** 2).clamp(min=1e-2)
		return debiased_mean, debiased_var

	def normalize(self, input_vector):
		if type(input_vector) == np.ndarray:
			input_vector = torch.from_numpy(input_vector)

		mean, var = self.debiased_mean_var()
		out = (input_vector - mean[(None,) * self.norm_axes]) / torch.sqrt(var)[(None,) * self.norm_axes]
		
		return out

	def denormalize(self, input_vector):
		if type(input_vector) == np.ndarray:
			input_vector = torch.from_numpy(input_vector)

		mean, var = self.debiased_mean_var()
		out = input_vector * torch.sqrt(var)[(None,) * self.norm_axes] + mean[(None,) * self.norm_axes]
		
		# out = out.cpu().numpy()

		return out


class Identity(nn.Module):
	def __init__(self):
		super(Identity, self).__init__()
		
	def forward(self, x):
		return x

# class Policy(nn.Module):
# 	def __init__(self, obs_input_dim, num_actions, num_agents, device):
# 		super(Policy, self).__init__()

# 		self.name = "MLP Policy"

# 		self.num_agents = num_agents
# 		self.num_actions = num_actions
# 		self.device = device
# 		self.Policy_MLP = nn.Sequential(
# 			nn.Linear(obs_input_dim, 128),
# 			nn.Tanh(),
# 			nn.Linear(128, 64),
# 			nn.Tanh(),
# 			nn.Linear(64, num_actions),
# 			nn.Softmax(dim=-1)
# 			)

# 		self.reset_parameters()

# 	def reset_parameters(self):
# 		gain = nn.init.calculate_gain('tanh')
# 		gain_last_layer = nn.init.calculate_gain('tanh', 0.01)

# 		nn.init.orthogonal_(self.Policy_MLP[0].weight, gain=gain)
# 		nn.init.orthogonal_(self.Policy_MLP[2].weight, gain=gain)
# 		nn.init.orthogonal_(self.Policy_MLP[4].weight, gain=gain_last_layer)


# 	def forward(self, local_observations):
# 		return self.Policy_MLP(local_observations)


class Policy(nn.Module):
	def __init__(self, obs_input_dim, num_actions, num_agents, device):
		super(Policy, self).__init__()

		self.name = "MLP Policy"

		self.num_agents = num_agents
		self.num_actions = num_actions
		self.device = device
		self.Policy_MLP = nn.Sequential(
			nn.Linear(obs_input_dim, 256),
			nn.Tanh(),
			nn.Linear(256, 256),
			nn.Tanh(),
			nn.Linear(256, 256),
			nn.Tanh(),
			nn.Linear(256, num_actions),
			nn.Softmax(dim=-1)
			)

		self.reset_parameters()

	def reset_parameters(self):
		gain = nn.init.calculate_gain('tanh', 0.1)
		gain_last_layer = nn.init.calculate_gain('tanh', 0.01)

		nn.init.orthogonal_(self.Policy_MLP[0].weight, gain=gain)
		nn.init.orthogonal_(self.Policy_MLP[2].weight, gain=gain)
		nn.init.orthogonal_(self.Policy_MLP[4].weight, gain=gain)
		nn.init.orthogonal_(self.Policy_MLP[6].weight, gain=gain_last_layer)


	def forward(self, local_observations):
		return self.Policy_MLP(local_observations)


# using Q network of MAAC
# class Q_network(nn.Module):
# 	def __init__(self, obs_input_dim, num_agents, num_actions, value_normalization, device):
# 		super(Q_network, self).__init__()
		
# 		self.num_agents = num_agents
# 		self.num_actions = num_actions
# 		self.device = device
# 		self.value_normalization = value_normalization

# 		obs_output_dim = 256
# 		obs_act_input_dim = obs_input_dim+self.num_actions
# 		obs_act_output_dim = 256
# 		curr_agent_output_dim = 128

# 		self.state_embed = nn.Sequential(
# 			nn.Linear(obs_input_dim, 256, bias=True), 
# 			nn.Tanh()
# 			)
# 		self.key = nn.Linear(256, obs_output_dim, bias=True)
# 		self.query = nn.Linear(256, obs_output_dim, bias=True)

# 		self.hard_attention = nn.Sequential(
# 			nn.Linear(obs_output_dim*2, 64),
# 			nn.Tanh(),
# 			nn.Linear(64, 2)
# 			)
		
# 		self.state_act_embed = nn.Sequential(
# 			nn.Linear(obs_act_input_dim, obs_act_output_dim, bias=True), 
# 			nn.Tanh()
# 			)
# 		self.attention_value = nn.Sequential(
# 			nn.Linear(obs_act_output_dim, 256, bias=True), 
# 			nn.Tanh()
# 			)

# 		self.curr_agent_state_embed = nn.Sequential(
# 			nn.Linear(obs_input_dim, curr_agent_output_dim, bias=True), 
# 			nn.Tanh()
# 			)

# 		# dimesion of key
# 		self.d_k = obs_output_dim

# 		# ********************************************************************************************************

# 		# ********************************************************************************************************
# 		final_input_dim = obs_act_output_dim + curr_agent_output_dim
# 		# FCN FINAL LAYER TO GET VALUES
# 		if value_normalization:
# 			self.final_value_layers = nn.Sequential(
# 				nn.Linear(final_input_dim, 128, bias=True), 
# 				nn.Tanh(),
# 				)
# 			self.pop_art = PopArt(128, self.num_actions, norm_axes=1, device=self.device)
# 		else:
# 			self.final_value_layers = nn.Sequential(
# 				nn.Linear(final_input_dim, 128, bias=True), 
# 				nn.Tanh(),
# 				nn.Linear(128, self.num_actions, bias=True)
# 				)
			
# 		# ********************************************************************************************************
# 		self.reset_parameters()


# 	def reset_parameters(self):
# 		"""Reinitialize learnable parameters."""
# 		gain = nn.init.calculate_gain('tanh')

# 		nn.init.orthogonal_(self.state_embed[0].weight, gain=gain)
# 		nn.init.orthogonal_(self.state_act_embed[0].weight, gain=gain)

# 		nn.init.orthogonal_(self.key.weight, gain=gain)
# 		nn.init.orthogonal_(self.query.weight, gain=gain)
# 		nn.init.orthogonal_(self.attention_value[0].weight, gain=gain)

# 		nn.init.orthogonal_(self.hard_attention[0].weight, gain=gain)
# 		nn.init.orthogonal_(self.hard_attention[2].weight, gain=gain)

# 		nn.init.orthogonal_(self.curr_agent_state_embed[0].weight, gain=gain)

# 		nn.init.orthogonal_(self.final_value_layers[0].weight, gain=gain)

# 		if self.value_normalization:
# 			nn.init.orthogonal_(self.pop_art.weight, gain=gain)
# 		else:
# 			nn.init.orthogonal_(self.final_value_layers[2].weight, gain=gain)


# 	def remove_self_loops(self, states_key):
# 		ret_states_keys = torch.zeros(states_key.shape[0],self.num_agents,self.num_agents-1,states_key.shape[-1])
# 		for i in range(self.num_agents):
# 			if i == 0:
# 				red_state = states_key[:,i,i+1:]
# 			elif i == self.num_agents-1:
# 				red_state = states_key[:,i,:i]
# 			else:
# 				red_state = torch.cat([states_key[:,i,:i],states_key[:,i,i+1:]], dim=-2)

# 			ret_states_keys[:,i] = red_state

# 		return ret_states_keys.to(self.device)

# 	def weight_assignment(self,weights):
# 		weights_new = torch.zeros(weights.shape[0], self.num_agents, self.num_agents).to(self.device)
# 		one = torch.ones(weights.shape[0],1).to(self.device)
# 		for i in range(self.num_agents):
# 			if i == 0:
# 				weight_vec = torch.cat([one,weights[:,i,:]], dim=-1)
# 			elif i == self.num_agents-1:
# 				weight_vec = torch.cat([weights[:,i,:],one], dim=-1)
# 			else:
# 				weight_vec = torch.cat([weights[:,i,:i],one,weights[:,i,i:]], dim=-1)

# 			weights_new[:,i] = weight_vec

# 		return weights_new


# 	def forward(self, states, policies, actions):
# 		states_query = states.unsqueeze(-2)
# 		states_key = states.unsqueeze(1).repeat(1,self.num_agents,1,1)
# 		actions_ = actions.unsqueeze(1).repeat(1,self.num_agents,1,1)

# 		states_key = self.remove_self_loops(states_key)
# 		actions_ = self.remove_self_loops(actions_)

# 		obs_actions = torch.cat([states_key,actions_],dim=-1)

# 		# EMBED STATES QUERY
# 		states_query_embed = self.state_embed(states_query)
# 		# EMBED STATES QUERY
# 		states_key_embed = self.state_embed(states_key)
# 		# KEYS
# 		key_obs = self.key(states_key_embed)
# 		# print(key_obs.shape)
# 		# QUERIES
# 		query_obs = self.query(states_query_embed)
# 		# print(query_obs.shape)
# 		# WEIGHT
# 		query_vector = torch.cat([query_obs.repeat(1,1,self.num_agents-1,1), key_obs], dim=-1)
# 		hard_weights = nn.functional.gumbel_softmax(self.hard_attention(query_vector), hard=True, dim=-1)
# 		prop = hard_weights[:,:,:,1]
# 		hard_score = -10000*(1-prop) + prop
# 		score = (torch.matmul(query_obs,key_obs.transpose(2,3))/math.sqrt(self.d_k)) + hard_score.unsqueeze(-2)
# 		weight = F.softmax(score ,dim=-1)
# 		weights = self.weight_assignment(weight.squeeze(-2))

# 		# EMBED STATE ACTION POLICY
# 		obs_actions_embed = self.state_act_embed(obs_actions)
# 		attention_values = self.attention_value(obs_actions_embed)
# 		node_features = torch.matmul(weight, attention_values)

# 		curr_agent_state_embed = self.curr_agent_state_embed(states)
# 		curr_agent_node_features = torch.cat([curr_agent_state_embed, node_features.squeeze(-2)], dim=-1)
		
# 		Q_value = self.final_value_layers(curr_agent_node_features)

# 		if self.value_normalization:
# 			Q_value = self.pop_art(Q_value)

# 		Value = torch.matmul(Q_value,policies.transpose(1,2))

# 		Q_value = torch.sum(actions*Q_value, dim=-1).unsqueeze(-1)

# 		return Value, Q_value, weights


class Q_network(nn.Module):
	def __init__(self, obs_input_dim, num_agents, num_actions, device):
		super(Q_network, self).__init__()
		
		self.num_agents = num_agents
		self.num_actions = num_actions
		self.device = device

		obs_output_dim = 256
		obs_act_input_dim = obs_input_dim+self.num_actions
		obs_act_output_dim = 256
		curr_agent_output_dim = 256

		self.state_embed = nn.Sequential(
			nn.Linear(obs_input_dim, 256, bias=True), 
			nn.Tanh(),
			nn.Linear(256, 256, bias=True), 
			nn.Tanh()
			)
		self.key = nn.Sequential(
			nn.Linear(256, 256, bias=True),
			nn.Tanh(),
			nn.Linear(256, obs_output_dim, bias=True),
			nn.Tanh(),
			)
		self.query = nn.Sequential(
			nn.Linear(256, 256, bias=True),
			nn.Tanh(),
			nn.Linear(256, obs_output_dim, bias=True),
			nn.Tanh(),
			)
		
		self.state_act_embed = nn.Sequential(
			nn.Linear(obs_act_input_dim, 256, bias=True), 
			nn.Tanh(),
			nn.Linear(256, obs_act_output_dim, bias=True), 
			nn.Tanh(),
			)
		self.attention_value = nn.Sequential(
			nn.Linear(obs_act_output_dim, 256, bias=True), 
			nn.Tanh(),
			nn.Linear(256, 256, bias=True), 
			nn.Tanh(),
			)
		self.layer_norm_state_act_embed = nn.LayerNorm(obs_act_output_dim)

		self.curr_agent_state_embed = nn.Sequential(
			nn.Linear(obs_input_dim, 256, bias=True), 
			nn.Tanh(),
			nn.Linear(256, curr_agent_output_dim, bias=True), 
			nn.Tanh(),
			)
		self.layer_norm_state_embed = nn.LayerNorm(curr_agent_output_dim)

		# dimesion of key
		self.d_k = obs_output_dim

		# ********************************************************************************************************

		# ********************************************************************************************************
		final_input_dim = obs_act_output_dim + curr_agent_output_dim
		# FCN FINAL LAYER TO GET VALUES
		self.final_value_layers = nn.Sequential(
			nn.Linear(final_input_dim, 256, bias=True), 
			nn.Tanh(),
			nn.Linear(256, 256, bias=True), 
			nn.Tanh(),
			nn.Linear(256, self.num_actions, bias=True)
			)
			
		# ********************************************************************************************************
		self.reset_parameters()


	def reset_parameters(self):
		"""Reinitialize learnable parameters."""
		gain = nn.init.calculate_gain('tanh', 0.1)

		# EMBEDDINGS
		nn.init.orthogonal_(self.state_embed[0].weight, gain=gain)
		nn.init.orthogonal_(self.state_embed[2].weight, gain=gain)
		nn.init.orthogonal_(self.state_act_embed[0].weight, gain=gain)
		nn.init.orthogonal_(self.state_act_embed[2].weight, gain=gain)

		nn.init.orthogonal_(self.key[0].weight, gain=gain)
		nn.init.orthogonal_(self.key[2].weight, gain=gain)
		nn.init.orthogonal_(self.query[0].weight, gain=gain)
		nn.init.orthogonal_(self.query[2].weight, gain=gain)
		nn.init.orthogonal_(self.attention_value[0].weight, gain=gain)
		nn.init.orthogonal_(self.attention_value[2].weight, gain=gain)

		nn.init.orthogonal_(self.curr_agent_state_embed[0].weight, gain=gain)
		nn.init.orthogonal_(self.curr_agent_state_embed[2].weight, gain=gain)

		nn.init.orthogonal_(self.final_value_layers[0].weight, gain=gain)
		nn.init.orthogonal_(self.final_value_layers[2].weight, gain=gain)
		nn.init.orthogonal_(self.final_value_layers[4].weight)


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

	def weight_assignment(self,weights):
		weights_new = torch.zeros(weights.shape[0], self.num_agents, self.num_agents).to(self.device)
		one = torch.ones(weights.shape[0],1).to(self.device)
		for i in range(self.num_agents):
			if i == 0:
				weight_vec = torch.cat([one,weights[:,i,:]], dim=-1)
			elif i == self.num_agents-1:
				weight_vec = torch.cat([weights[:,i,:],one], dim=-1)
			else:
				weight_vec = torch.cat([weights[:,i,:i],one,weights[:,i,i:]], dim=-1)

			weights_new[:,i] = weight_vec

		return weights_new


	def forward(self, states, policies, actions):
		states_query = states.unsqueeze(-2)
		states_key = states.unsqueeze(1).repeat(1,self.num_agents,1,1)
		actions_ = actions.unsqueeze(1).repeat(1,self.num_agents,1,1)

		states_key = self.remove_self_loops(states_key)
		actions_ = self.remove_self_loops(actions_)

		obs_actions = torch.cat([states_key,actions_],dim=-1)

		# EMBED STATES QUERY
		states_query_embed = self.state_embed(states_query)
		# EMBED STATES QUERY
		states_key_embed = self.state_embed(states_key)
		# KEYS
		key_obs = self.key(states_key_embed)
		# QUERIES
		query_obs = self.query(states_query_embed)
		# WEIGHT
		scores = torch.matmul(query_obs,key_obs.transpose(2,3))/math.sqrt(self.d_k)
		weight = F.softmax(scores,dim=-1)
		weights = self.weight_assignment(weight.squeeze(-2))

		# EMBED STATE ACTION POLICY
		obs_actions_embed = self.state_act_embed(obs_actions)
		attention_values = self.attention_value(obs_actions_embed)
		node_features = torch.matmul(weight, attention_values)
		node_features = self.layer_norm_state_act_embed(torch.mean(obs_actions_embed, dim=-2).unsqueeze(-2)+node_features)

		curr_agent_state_embed = self.curr_agent_state_embed(states)
		curr_agent_state_embed = self.layer_norm_state_embed(curr_agent_state_embed+torch.mean(states_key_embed, dim=-2).squeeze(-2)+states_query_embed.squeeze(-2))
		curr_agent_node_features = torch.cat([curr_agent_state_embed, node_features.squeeze(-2)], dim=-1)
		
		Q_value = self.final_value_layers(curr_agent_node_features)

		Value = torch.matmul(Q_value,policies.transpose(1,2))

		Q_value = torch.sum(actions*Q_value, dim=-1).unsqueeze(-1)

		return Value, Q_value, weights