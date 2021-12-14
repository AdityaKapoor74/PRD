from typing import Any, List, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import datetime
import math


class MLPPolicy(nn.Module):
	def __init__(self,state_dim,num_agents,action_dim):
		super(MLPPolicy,self).__init__()

		self.name = "MLPPolicy"

		self.state_dim = state_dim
		self.num_agents = num_agents		
		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		# self.device = "cpu"


		self.fc1 = nn.Linear(state_dim*num_agents,128)
		self.fc2 = nn.Linear(128,128)
		self.fc3 = nn.Linear(128,action_dim)

	def forward(self, states):

		# T x num_agents x state_dim
		T = states.shape[0]
		
		# [s0;s1;s2;s3]  -> [s0 s1 s2 s3; s1 s2 s3 s0; s2 s3 s1 s0 ....]

		states_aug = [torch.roll(states,i,1) for i in range(self.num_agents)]

		states_aug = torch.cat(states_aug,dim=2)

		x = self.fc1(states_aug)
		x = nn.ReLU()(x)
		x = self.fc2(x)
		x = nn.ReLU()(x)
		x = self.fc3(x)

		Policy = F.softmax(x, dim=-1)

		return Policy, 1/self.num_agents*torch.ones((T,self.num_agents,self.num_agents),device=self.device)


# class TransformerPolicy(nn.Module):
# 	def __init__(self, obs_input_dim, obs_output_dim, final_input_dim, final_output_dim, num_agents, num_actions):
# 		super(TransformerPolicy, self).__init__()
		
# 		self.name = "TransformerPolicy"

# 		self.num_agents = num_agents
# 		self.num_actions = num_actions
# 		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 		# self.device = "cpu"

# 		self.state_embed = nn.Sequential(nn.Linear(obs_input_dim, 128), nn.LeakyReLU())
# 		self.key_layer = nn.Linear(128, obs_output_dim, bias=False)
# 		self.query_layer = nn.Linear(128, obs_output_dim, bias=False)
# 		self.attention_value_layer = nn.Linear(128, obs_output_dim, bias=False)
# 		# dimesion of key
# 		self.d_k_obs_act = obs_output_dim  

# 		# NOISE
# 		self.noise_normal = torch.distributions.Normal(loc=torch.tensor([0.0]), scale=torch.tensor([1.0]))
# 		self.noise_uniform = torch.rand
# 		# ********************************************************************************************************

# 		# ********************************************************************************************************
# 		# FCN FINAL LAYER TO GET VALUES
# 		self.final_policy_layer_1 = nn.Linear(final_input_dim, 64, bias=False)
# 		self.final_policy_layer_2 = nn.Linear(64, final_output_dim, bias=False)
# 		# ********************************************************************************************************


# 		self.reset_parameters()


# 	def reset_parameters(self):
# 		"""Reinitialize learnable parameters."""
# 		gain_leaky = nn.init.calculate_gain('leaky_relu')

# 		nn.init.xavier_uniform_(self.state_embed[0].weight, gain=gain_leaky)

# 		nn.init.xavier_uniform_(self.key_layer.weight)
# 		nn.init.xavier_uniform_(self.query_layer.weight)
# 		nn.init.xavier_uniform_(self.attention_value_layer.weight)


# 		nn.init.xavier_uniform_(self.final_policy_layer_1.weight, gain=gain_leaky)
# 		nn.init.xavier_uniform_(self.final_policy_layer_2.weight, gain=gain_leaky)



# 	def forward(self, states):
# 		# EMBED STATES
# 		states_embed = self.state_embed(states)
# 		# KEYS
# 		key_obs = self.key_layer(states_embed)
# 		# QUERIES
# 		query_obs = self.query_layer(states_embed)
# 		# WEIGHT
# 		weight = F.softmax(torch.matmul(query_obs,key_obs.transpose(1,2))/math.sqrt(self.d_k_obs_act),dim=-1)
# 		attention_values = self.attention_value_layer(states_embed)
# 		node_features = torch.matmul(weight, attention_values)


# 		Policy = F.leaky_relu(self.final_policy_layer_1(node_features))
# 		Policy = F.softmax(self.final_policy_layer_2(Policy), dim=-1)

# 		return Policy, weight


'''
Scalar Dot Product Attention: Transformer
'''
class TransformerCritic(nn.Module):
	'''
	https://proceedings.neurips.cc/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf
	'''
	def __init__(self, obs_input_dim, obs_output_dim, obs_act_input_dim, obs_act_output_dim, final_input_dim, final_output_dim, num_agents, num_actions):
		super(TransformerCritic, self).__init__()
		
		self.name = "TransformerCritic"

		self.num_agents = num_agents
		self.num_actions = num_actions
		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		# self.device = "cpu"

		self.state_embed = nn.Sequential(nn.Linear(obs_input_dim, 128), nn.LeakyReLU())
		self.key_layer = nn.Linear(128, obs_output_dim, bias=False)
		self.query_layer = nn.Linear(128, obs_output_dim, bias=False)
		self.state_act_embed = nn.Sequential(nn.Linear(obs_act_input_dim, 128), nn.LeakyReLU())
		self.attention_value_layer = nn.Linear(128, obs_act_output_dim, bias=False)
		# dimesion of key
		self.d_k_obs_act = obs_output_dim  
		# ********************************************************************************************************
		# EMBED S of agent whose Q value is being est
		self.state_embed_q = nn.Sequential(nn.Linear(obs_input_dim, obs_output_dim), nn.LeakyReLU())
		# ********************************************************************************************************
		# FCN FINAL LAYER TO GET VALUES
		self.final_value_layers = nn.Sequential(
												nn.Linear(final_input_dim, 64, bias=False),
												nn.LeakyReLU(),
												nn.Linear(64, final_output_dim, bias=False)
												)
		# ********************************************************************************************************	


		self.reset_parameters()


	def reset_parameters(self):
		"""Reinitialize learnable parameters."""
		gain_leaky = nn.init.calculate_gain('leaky_relu')

		nn.init.xavier_uniform_(self.state_embed[0].weight, gain=gain_leaky)
		nn.init.xavier_uniform_(self.state_embed_q[0].weight, gain=gain_leaky)
		nn.init.xavier_uniform_(self.state_act_embed[0].weight, gain=gain_leaky)

		nn.init.xavier_uniform_(self.key_layer.weight)
		nn.init.xavier_uniform_(self.query_layer.weight)
		nn.init.xavier_uniform_(self.attention_value_layer.weight)


		nn.init.xavier_uniform_(self.final_value_layers[0].weight, gain=gain_leaky)
		nn.init.xavier_uniform_(self.final_value_layers[2].weight, gain=gain_leaky)



	def forward(self, states, actions):
		# EMBED STATES
		states_embed = self.state_embed(states)
		# KEYS
		key_obs = self.key_layer(states_embed)
		# QUERIES
		query_obs = self.query_layer(states_embed)
		# WEIGHT
		weight = F.softmax(torch.matmul(query_obs,key_obs.transpose(1,2))/math.sqrt(self.d_k_obs_act),dim=-1)

		obs_actions = torch.cat([states,actions],dim=-1)
		# EMBED STATE ACTION POLICY
		obs_actions_embed = self.state_act_embed(obs_actions)
		attention_values = self.attention_value_layer(obs_actions_embed)

		agg_attention_values = torch.matmul(weight, attention_values)
		state_embed_q = self.state_embed_q(states)
		node_features = torch.cat([state_embed_q, agg_attention_values], dim=-1)

		Value = self.final_value_layers(node_features)

		return Value, weight


class DualTransformerCritic(nn.Module):
	def __init__(self, obs_input_dim, obs_output_dim, obs_act_input_dim, obs_act_output_dim, final_input_dim, final_output_dim, num_agents, num_actions):
		super(DualTransformerCritic, self).__init__()
		
		self.name = "DualTransformerCritic"

		self.num_agents = num_agents
		self.num_actions = num_actions
		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		# self.device = "cpu"

		self.state_embed_preproc = nn.Sequential(nn.Linear(obs_input_dim, 128), nn.LeakyReLU())
		self.key_layer_preproc = nn.Linear(128, obs_output_dim, bias=False)
		self.query_layer_preproc = nn.Linear(128, obs_output_dim, bias=False)
		self.attention_value_layer_preproc = nn.Linear(128, obs_output_dim, bias=False)
		self.d_k_obs = obs_output_dim

		self.state_embed = nn.Sequential(nn.Linear(obs_output_dim, 128), nn.LeakyReLU())
		self.key_layer = nn.Linear(128, obs_output_dim, bias=False)
		self.query_layer = nn.Linear(128, obs_output_dim, bias=False)
		self.state_act_embed = nn.Sequential(nn.Linear(obs_act_input_dim, 128), nn.LeakyReLU())
		self.attention_value_layer = nn.Linear(128, obs_act_output_dim, bias=False)
		# dimesion of key
		self.d_k_obs_act = obs_output_dim  

		# EMBED S of agent whose Q value is being est
		self.state_embed_q = nn.Sequential(nn.Linear(obs_output_dim, obs_output_dim), nn.LeakyReLU())

		self.final_value_layers = nn.Sequential(
												nn.Linear(final_input_dim, 64, bias=False),
												nn.LeakyReLU(),
												nn.Linear(64, final_output_dim, bias=False)
												)

		self.reset_parameters()


	def reset_parameters(self):
		"""Reinitialize learnable parameters."""
		gain_leaky = nn.init.calculate_gain('leaky_relu')

		nn.init.xavier_uniform_(self.state_embed_preproc[0].weight, gain=gain_leaky)

		nn.init.xavier_uniform_(self.key_layer_preproc.weight)
		nn.init.xavier_uniform_(self.query_layer_preproc.weight)
		nn.init.xavier_uniform_(self.attention_value_layer_preproc.weight)

		nn.init.xavier_uniform_(self.state_embed[0].weight, gain=gain_leaky)
		nn.init.xavier_uniform_(self.state_embed_q[0].weight, gain=gain_leaky)
		nn.init.xavier_uniform_(self.state_act_embed[0].weight, gain=gain_leaky)

		nn.init.xavier_uniform_(self.key_layer.weight)
		nn.init.xavier_uniform_(self.query_layer.weight)
		nn.init.xavier_uniform_(self.attention_value_layer.weight)


		nn.init.xavier_uniform_(self.final_value_layers[0].weight, gain=gain_leaky)
		nn.init.xavier_uniform_(self.final_value_layers[2].weight, gain=gain_leaky)



	def forward(self, states, actions):
		# EMBED STATES PREPROC
		states_embed_preproc = self.state_embed_preproc(states)
		# KEYS
		key_obs_preproc = self.key_layer(states_embed_preproc)
		# QUERIES
		query_obs_preproc = self.query_layer(states_embed_preproc)
		# WEIGHT
		weight_preproc = F.softmax(torch.matmul(query_obs_preproc,key_obs_preproc.transpose(1,2))/math.sqrt(self.d_k_obs),dim=-1)
		# ATTENTION VALUES
		attention_values_preproc = self.attention_value_layer_preproc(states_embed_preproc)
		attention_values_preproc = torch.matmul(weight_preproc, states_embed_preproc)


		# EMBED STATES
		states_embed = self.state_embed(attention_values_preproc)
		# KEYS
		key_obs = self.key_layer(states_embed)
		# QUERIES
		query_obs = self.query_layer(states_embed)
		# WEIGHT
		weight_post_proc = F.softmax(torch.matmul(query_obs,key_obs.transpose(1,2))/math.sqrt(self.d_k_obs_act),dim=-1)

		obs_actions = torch.cat([states,actions],dim=-1)
		# EMBED STATE ACTION POLICY
		obs_actions_embed = self.state_act_embed(obs_actions)
		attention_values = self.attention_value_layer(obs_actions_embed)

		agg_attention_values = torch.matmul(weight_post_proc, attention_values)
		state_embed_q = self.state_embed_q(attention_values_preproc)

		node_features = torch.cat([state_embed_q, agg_attention_values], dim=-1)

		Value = self.final_value_layers(node_features)

		return Value, weight_preproc, weight_post_proc


# '''
# Multi Agent Actor Attention Critic (Q Network inspired from them)
# '''
# class TransformerCritic(nn.Module):
# 	'''
# 	https://proceedings.neurips.cc/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf
# 	'''
# 	def __init__(self, obs_input_dim, obs_output_dim, obs_act_input_dim, obs_act_output_dim, final_input_dim, final_output_dim, num_agents, num_actions):
# 		super(TransformerCritic, self).__init__()
		
# 		self.name = "TransformerCritic"

# 		self.num_agents = num_agents
# 		self.num_actions = num_actions
# 		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 		self.state_act_embed_attn = nn.Sequential(nn.Linear(obs_act_input_dim, 128), nn.LeakyReLU())
# 		self.state_embed_attn = nn.Sequential(nn.Linear(obs_input_dim, 128), nn.LeakyReLU())
# 		self.key_layer = nn.Linear(128, obs_output_dim, bias=False)
# 		self.query_layer = nn.Linear(128, obs_output_dim, bias=False)
# 		self.attention_value_layer = nn.Linear(128, obs_act_output_dim, bias=False)
# 		# dimesion of key
# 		self.d_k_obs = obs_output_dim  

# 		# score corresponding to current agent should be 0
# 		self.mask_score = torch.ones(self.num_agents,self.num_agents, dtype=torch.bool).to(self.device)
# 		self.mask_attn_values = torch.ones(self.num_agents,self.num_agents, 128, dtype=torch.bool).to(self.device)
# 		mask = torch.zeros(128, dtype=torch.bool).to(self.device)
# 		for j in range(self.num_agents):
# 			self.mask_score[j][j] = False
# 			self.mask_attn_values[j][j] = mask

# 		# ********************************************************************************************************

# 		# ********************************************************************************************************
# 		# EMBED S of agent whose Q value is being est
# 		self.state_embed_q = nn.Sequential(nn.Linear(obs_input_dim, obs_output_dim), nn.LeakyReLU())
# 		# FCN FINAL LAYER TO GET VALUES
# 		self.final_value_layers = nn.Sequential(
# 												nn.Linear(final_input_dim, 64, bias=False),
# 												nn.LeakyReLU(),
# 												nn.Linear(64, final_output_dim, bias=False)
# 												)
# 		# ********************************************************************************************************
# 		self.reset_parameters()


# 	def reset_parameters(self):
# 		"""Reinitialize learnable parameters."""
# 		gain_leaky = nn.init.calculate_gain('leaky_relu')

# 		nn.init.xavier_uniform_(self.state_act_embed_attn[0].weight, gain=gain_leaky)
# 		nn.init.xavier_uniform_(self.state_embed_attn[0].weight, gain=gain_leaky)
# 		nn.init.xavier_uniform_(self.state_embed_q[0].weight, gain=gain_leaky)

# 		nn.init.xavier_uniform_(self.key_layer.weight)
# 		nn.init.xavier_uniform_(self.query_layer.weight)
# 		nn.init.xavier_uniform_(self.attention_value_layer.weight)

# 		nn.init.xavier_uniform_(self.final_value_layers[0].weight, gain=gain_leaky)
# 		nn.init.xavier_uniform_(self.final_value_layers[2].weight, gain=gain_leaky)


# 	def forward(self, states, actions):
# 		state_actions = torch.cat([states, actions], dim=-1)
# 		state_embed_attn = self.state_embed_attn(states)
# 		state_act_embed_attn = self.state_act_embed_attn(state_actions)
# 		# Keys
# 		keys = self.key_layer(state_embed_attn)
# 		# Queries
# 		queries = self.query_layer(state_embed_attn)
# 		# Calc score (score corresponding to self to be made 0)
# 		score = torch.matmul(queries,keys.transpose(1,2))/math.sqrt(self.d_k_obs)
# 		mask = torch.ones_like(score, dtype=torch.bool).to(self.device)*self.mask_score
# 		score = score[mask].reshape(mask.shape[0], self.num_agents,-1)
# 		weight = F.softmax(score,dim=-1)
# 		attention_values = self.attention_value_layer(state_act_embed_attn).unsqueeze(1).repeat(1,self.num_agents,1,1)
# 		mask_attn = torch.ones_like(attention_values, dtype=torch.bool).to(self.device) * self.mask_attn_values
# 		attention_values = attention_values[mask_attn].reshape(mask.shape[0], self.num_agents, self.num_agents-1,-1)
# 		x = torch.sum(attention_values*weight.unsqueeze(-1), dim=-2)

# 		# Embedding state of current agent
# 		curr_agent_state_action_embed = self.state_embed_q(states)

# 		node_features = torch.cat([curr_agent_state_action_embed, x], dim=-1)

# 		Value = self.final_value_layers(node_features)

# 		return Value, weight



# class DualTransformerCritic(nn.Module):
# 	def __init__(self, obs_input_dim, obs_output_dim, obs_act_input_dim, obs_act_output_dim, final_input_dim, final_output_dim, num_agents, num_actions):
# 		super(DualTransformerCritic, self).__init__()
		
# 		self.name = "DualTransformerCritic"

# 		self.num_agents = num_agents
# 		self.num_actions = num_actions
# 		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 		# self.device = "cpu"

# 		self.state_embed_preproc = nn.Sequential(nn.Linear(obs_input_dim, 128), nn.LeakyReLU())
# 		self.key_layer_preproc = nn.Linear(128, obs_output_dim, bias=False)
# 		self.query_layer_preproc = nn.Linear(128, obs_output_dim, bias=False)
# 		self.attention_value_layer_preproc = nn.Linear(128, obs_output_dim, bias=False)
# 		self.d_k_obs = obs_output_dim


# 		self.state_act_embed_attn = nn.Sequential(nn.Linear(obs_act_input_dim, 128), nn.LeakyReLU())
# 		self.state_embed_attn = nn.Sequential(nn.Linear(obs_output_dim, 128), nn.LeakyReLU())
# 		self.key_layer = nn.Linear(128, obs_output_dim, bias=False)
# 		self.query_layer = nn.Linear(128, obs_output_dim, bias=False)
# 		self.attention_value_layer = nn.Linear(128, obs_act_output_dim, bias=False)
# 		# dimesion of key
# 		self.d_k_obs_ = obs_output_dim  

# 		# score corresponding to current agent should be 0
# 		self.mask_score = torch.ones(self.num_agents,self.num_agents, dtype=torch.bool).to(self.device)
# 		self.mask_attn_values = torch.ones(self.num_agents,self.num_agents, 128, dtype=torch.bool).to(self.device)
# 		mask = torch.zeros(128, dtype=torch.bool).to(self.device)
# 		for j in range(self.num_agents):
# 			self.mask_score[j][j] = False
# 			self.mask_attn_values[j][j] = mask
# 		# ********************************************************************************************************

# 		# ********************************************************************************************************
# 		# EMBED S of agent whose Q value is being est
# 		self.state_embed_q = nn.Sequential(nn.Linear(obs_output_dim, obs_output_dim), nn.LeakyReLU())
# 		# FCN FINAL LAYER TO GET VALUES
# 		self.final_value_layers = nn.Sequential(
# 												nn.Linear(final_input_dim, 64, bias=False),
# 												nn.LeakyReLU(),
# 												nn.Linear(64, final_output_dim, bias=False)
# 												)
# 		# ********************************************************************************************************
# 		self.reset_parameters()


# 	def reset_parameters(self):
# 		"""Reinitialize learnable parameters."""
# 		gain_leaky = nn.init.calculate_gain('leaky_relu')

# 		nn.init.xavier_uniform_(self.state_embed_preproc[0].weight, gain=gain_leaky)

# 		nn.init.xavier_uniform_(self.key_layer_preproc.weight)
# 		nn.init.xavier_uniform_(self.query_layer_preproc.weight)
# 		nn.init.xavier_uniform_(self.attention_value_layer_preproc.weight)

# 		nn.init.xavier_uniform_(self.state_act_embed_attn[0].weight, gain=gain_leaky)
# 		nn.init.xavier_uniform_(self.state_embed_attn[0].weight, gain=gain_leaky)
# 		nn.init.xavier_uniform_(self.state_embed_q[0].weight, gain=gain_leaky)

# 		nn.init.xavier_uniform_(self.key_layer.weight)
# 		nn.init.xavier_uniform_(self.query_layer.weight)
# 		nn.init.xavier_uniform_(self.attention_value_layer.weight)

# 		nn.init.xavier_uniform_(self.final_value_layers[0].weight, gain=gain_leaky)
# 		nn.init.xavier_uniform_(self.final_value_layers[2].weight, gain=gain_leaky)



# 	def forward(self, states, actions):
# 		# EMBED STATES PREPROC
# 		states_embed_preproc = self.state_embed_preproc(states)
# 		# KEYS
# 		key_obs_preproc = self.key_layer(states_embed_preproc)
# 		# QUERIES
# 		query_obs_preproc = self.query_layer(states_embed_preproc)
# 		# WEIGHT
# 		weight_preproc = F.softmax(torch.matmul(query_obs_preproc,key_obs_preproc.transpose(1,2))/math.sqrt(self.d_k_obs),dim=-1)
# 		# ATTENTION VALUES
# 		attention_values_preproc = self.attention_value_layer_preproc(states_embed_preproc)
# 		attention_values_preproc = torch.matmul(weight_preproc, states_embed_preproc)


# 		state_actions = torch.cat([attention_values_preproc, actions], dim=-1)
# 		state_embed_attn = self.state_embed_attn(attention_values_preproc)
# 		state_act_embed_attn = self.state_act_embed_attn(state_actions)
# 		# Keys
# 		keys = self.key_layer(state_embed_attn)
# 		# Queries
# 		queries = self.query_layer(state_embed_attn)
# 		# Calc score (score corresponding to self to be made 0)
# 		score = torch.matmul(queries,keys.transpose(1,2))/math.sqrt(self.d_k_obs_)
# 		mask = torch.ones_like(score, dtype=torch.bool).to(self.device)*self.mask_score
# 		score = score[mask].reshape(mask.shape[0], self.num_agents,-1)
# 		weight_post_proc = F.softmax(score,dim=-1)
# 		attention_values = self.attention_value_layer(state_act_embed_attn).unsqueeze(1).repeat(1,self.num_agents,1,1)
# 		mask_attn = torch.ones_like(attention_values, dtype=torch.bool).to(self.device) * self.mask_attn_values
# 		attention_values = attention_values[mask_attn].reshape(mask.shape[0], self.num_agents, self.num_agents-1,-1)
# 		x = torch.sum(attention_values*weight_post_proc.unsqueeze(-1), dim=-2)

# 		# Embedding state of current agent
# 		curr_agent_state_action_embed = self.state_embed_q(attention_values_preproc)

# 		node_features = torch.cat([curr_agent_state_action_embed, x], dim=-1)

# 		Value = self.final_value_layers(node_features)

# 		return Value, weight_preproc, weight_post_proc
# '''