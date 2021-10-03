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


class TransformerPolicy(nn.Module):
	def __init__(self, obs_input_dim, obs_output_dim, final_input_dim, final_output_dim, num_agents, num_actions):
		super(TransformerPolicy, self).__init__()
		
		self.name = "TransformerPolicy"

		self.num_agents = num_agents
		self.num_actions = num_actions
		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		# self.device = "cpu"

		self.state_embed = nn.Sequential(nn.Linear(obs_input_dim, 128), nn.LeakyReLU())
		self.key_layer = nn.Linear(128, obs_output_dim, bias=False)
		self.query_layer = nn.Linear(128, obs_output_dim, bias=False)
		self.attention_value_layer = nn.Linear(128, obs_output_dim, bias=False)
		# dimesion of key
		self.d_k_obs_act = obs_output_dim  

		# NOISE
		self.noise_normal = torch.distributions.Normal(loc=torch.tensor([0.0]), scale=torch.tensor([1.0]))
		self.noise_uniform = torch.rand
		# ********************************************************************************************************

		# ********************************************************************************************************
		# FCN FINAL LAYER TO GET VALUES
		self.final_policy_layer_1 = nn.Linear(final_input_dim, 64, bias=False)
		self.final_policy_layer_2 = nn.Linear(64, final_output_dim, bias=False)
		# ********************************************************************************************************


		self.reset_parameters()


	def reset_parameters(self):
		"""Reinitialize learnable parameters."""
		gain_leaky = nn.init.calculate_gain('leaky_relu')

		nn.init.xavier_uniform_(self.state_embed[0].weight, gain=gain_leaky)

		nn.init.xavier_uniform_(self.key_layer.weight)
		nn.init.xavier_uniform_(self.query_layer.weight)
		nn.init.xavier_uniform_(self.attention_value_layer.weight)


		nn.init.xavier_uniform_(self.final_policy_layer_1.weight, gain=gain_leaky)
		nn.init.xavier_uniform_(self.final_policy_layer_2.weight, gain=gain_leaky)



	def forward(self, states):
		# EMBED STATES
		states_embed = self.state_embed(states)
		# KEYS
		key_obs = self.key_layer(states_embed)
		# QUERIES
		query_obs = self.query_layer(states_embed)
		# WEIGHT
		weight = F.softmax(torch.matmul(query_obs,key_obs.transpose(1,2))/math.sqrt(self.d_k_obs_act),dim=-1)
		attention_values = self.attention_value_layer(states_embed)
		node_features = torch.matmul(weight, attention_values)


		Policy = F.leaky_relu(self.final_policy_layer_1(node_features))
		Policy = F.softmax(self.final_policy_layer_2(Policy), dim=-1)

		return Policy, weight


'''
Scalar Dot Product Attention
'''

class TransformerStateTransformerStateAction(nn.Module):
	'''
	https://proceedings.neurips.cc/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf
	'''
	def __init__(self, obs_input_dim, obs_output_dim, obs_act_input_dim, obs_act_output_dim, state_value_input_dim, state_value_output_dim, state_action_value_input_dim, state_action_value_output_dim num_agents, num_actions):
		super(TransformerStateTransformerStateAction, self).__init__()
		
		self.name = "TransformerStateTransformerStateAction"

		self.num_agents = num_agents
		self.num_actions = num_actions
		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		# self.device = "cpu"

		self.state_embed_transformer1 = nn.Sequential(nn.Linear(obs_input_dim, 128), nn.LeakyReLU())
		self.state_embed_transformer2 = nn.Sequential(nn.Linear(obs_input_dim, 128), nn.LeakyReLU())
		self.state_act_pol_embed = nn.Sequential(nn.Linear(obs_act_input_dim, 128), nn.LeakyReLU())

		self.key_layer_transformer1 = nn.Linear(128, obs_output_dim, bias=False)
		self.query_layer_transformer1 = nn.Linear(128, obs_output_dim, bias=False)
		self.attention_value_layer_transformer1 = nn.Linear(128, obs_output_dim, bias=False)

		self.key_layer_transformer2 = nn.Linear(128, obs_output_dim, bias=False)
		self.query_layer_transformer2 = nn.Linear(128, obs_output_dim, bias=False)
		self.attention_value_layer_transformer2 = nn.Linear(128, obs_act_output_dim, bias=False)


		# dimesion of key
		self.d_k_transformer1 = obs_output_dim 
		self.d_k_transformer2 = obs_output_dim 

		# NOISE
		self.noise_normal = torch.distributions.Normal(loc=torch.tensor([0.0]), scale=torch.tensor([1.0]))
		self.noise_uniform = torch.rand
		# ********************************************************************************************************

		# ********************************************************************************************************
		# FCN FINAL LAYER TO GET VALUES
		self.state_value_layer_1 = nn.Linear(state_value_input_dim, 64, bias=False)
		self.state_value_layer_2 = nn.Linear(64, state_value_output_dim, bias=False)
		# FCN FINAL LAYER TO GET VALUES
		self.state_action_value_layer_1 = nn.Linear(state_action_value_input_dim, 64, bias=False)
		self.state_action_value_layer_2 = nn.Linear(64, state_action_value_output_dim, bias=False)
		# ********************************************************************************************************	

		self.place_policies = torch.zeros(self.num_agents,self.num_agents,obs_act_input_dim).to(self.device)
		self.place_actions = torch.ones(self.num_agents,self.num_agents,obs_act_input_dim).to(self.device)
		one_hots = torch.ones(obs_act_input_dim)
		zero_hots = torch.zeros(obs_act_input_dim)

		for j in range(self.num_agents):
			self.place_policies[j][j] = one_hots
			self.place_actions[j][j] = zero_hots


		self.reset_parameters()


	def reset_parameters(self):
		"""Reinitialize learnable parameters."""
		gain_leaky = nn.init.calculate_gain('leaky_relu')

		nn.init.xavier_uniform_(self.state_embed_transformer1[0].weight, gain=gain_leaky)
		nn.init.xavier_uniform_(self.state_embed_transformer2[0].weight, gain=gain_leaky)
		nn.init.xavier_uniform_(self.state_act_pol_embed[0].weight, gain=gain_leaky)

		nn.init.xavier_uniform_(self.key_layer_transformer1.weight)
		nn.init.xavier_uniform_(self.query_layer_transformer1.weight)
		nn.init.xavier_uniform_(self.attention_value_layer_transformer1.weight)

		nn.init.xavier_uniform_(self.key_layer_transformer2.weight)
		nn.init.xavier_uniform_(self.query_layer_transformer2.weight)
		nn.init.xavier_uniform_(self.attention_value_layer_transformer2.weight)

		nn.init.xavier_uniform_(self.state_value_layer_1.weight, gain=gain_leaky)
		nn.init.xavier_uniform_(self.state_value_layer_2.weight, gain=gain_leaky)

		nn.init.xavier_uniform_(self.state_action_value_layer_1.weight, gain=gain_leaky)
		nn.init.xavier_uniform_(self.state_action_value_layer_2.weight, gain=gain_leaky)




	def forward(self, states, policies, actions):
		# EMBED STATES
		states_embed_transformer1 = self.state_embed_transformer1(states)
		states_embed_transformer2 = self.state_embed_transformer1(states)
		obs_actions = torch.cat([states,actions],dim=-1)
		obs_policy = torch.cat([states,policies], dim=-1)
		obs_actions = obs_actions.repeat(1,self.num_agents,1).reshape(obs_actions.shape[0],self.num_agents,self.num_agents,-1)
		obs_policy = obs_policy.repeat(1,self.num_agents,1).reshape(obs_policy.shape[0],self.num_agents,self.num_agents,-1)
		obs_actions_policies = self.place_policies*obs_policy + self.place_actions*obs_actions
		# EMBED STATE ACTION POLICY
		obs_actions_policies_embed = self.state_act_pol_embed(obs_actions_policies)

		# TRANSFORMER OVER STATES
		# KEYS
		key_obs = self.key_layer_transformer1(states_embed_transformer1)
		# QUERIES
		query_obs = self.query_layer_transformer1(states_embed_transformer1)
		# WEIGHT
		weight_obs = F.softmax(torch.matmul(query_obs,key_obs.transpose(1,2))/math.sqrt(self.d_k_transformer1),dim=-1)
		attention_values_obs = self.attention_value_layer_transformer1(states_embed_transformer1)
		node_features_transformer1 = torch.matmul(weight_obs, attention_values_obs)
		state_value = F.leaky_relu(self.state_value_layer_1(node_features_transformer1))
		state_value = self.state_value_layer_2(state_value)


		# TRANSFORMER OVER STATES
		# KEYS
		key_obs_act = self.key_layer_transformer1(states_embed_transformer1)
		# QUERIES
		query_obs_act = self.query_layer_transformer1(states_embed_transformer1)
		# WEIGHT
		weight_obs_act = F.softmax(torch.matmul(query_obs_act,key_obs_act.transpose(1,2))/math.sqrt(self.d_k_transformer2),dim=-1)
		attention_values_obs_act = self.attention_value_layer_transformer2(obs_actions_policies_embed)
		attention_values_obs_act = attention_values_obs_act.repeat(1,self.num_agents,1,1).reshape(attention_values_obs_act.shape[0],self.num_agents,self.num_agents,self.num_agents,-1)
		ret_weight_obs_act = weight_obs_act
		weight_obs_act = weight_obs_act.unsqueeze(-2).repeat(1,1,self.num_agents,1).unsqueeze(-1)
		weighted_attention_values_state_actions = attention_values_obs_act*weight_obs_act
		node_features_transformer2 = torch.sum(weighted_attention_values_state_actions, dim=-2)
		state_action_value = F.leaky_relu(self.state_action_value_layer_1(node_features_transformer2))
		state_action_value = self.state_action_value_layer_2(state_action_value)

		Value = state_value + state_action_value

		return Value, weight_obs, ret_weight_obs_act



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
		self.state_act_pol_embed = nn.Sequential(nn.Linear(obs_act_input_dim, 128), nn.LeakyReLU())
		self.attention_value_layer = nn.Linear(128, obs_act_output_dim, bias=False)
		# dimesion of key
		self.d_k_obs_act = obs_output_dim  

		# NOISE
		self.noise_normal = torch.distributions.Normal(loc=torch.tensor([0.0]), scale=torch.tensor([1.0]))
		self.noise_uniform = torch.rand
		# ********************************************************************************************************

		# ********************************************************************************************************
		# FCN FINAL LAYER TO GET VALUES
		self.final_value_layer_1 = nn.Linear(final_input_dim, 64, bias=False)
		self.final_value_layer_2 = nn.Linear(64, final_output_dim, bias=False)
		# ********************************************************************************************************	

		self.place_policies = torch.zeros(self.num_agents,self.num_agents,obs_act_input_dim).to(self.device)
		self.place_actions = torch.ones(self.num_agents,self.num_agents,obs_act_input_dim).to(self.device)
		one_hots = torch.ones(obs_act_input_dim)
		zero_hots = torch.zeros(obs_act_input_dim)

		for j in range(self.num_agents):
			self.place_policies[j][j] = one_hots
			self.place_actions[j][j] = zero_hots


		self.reset_parameters()


	def reset_parameters(self):
		"""Reinitialize learnable parameters."""
		gain_leaky = nn.init.calculate_gain('leaky_relu')

		nn.init.xavier_uniform_(self.state_embed[0].weight, gain=gain_leaky)
		nn.init.xavier_uniform_(self.state_act_pol_embed[0].weight, gain=gain_leaky)

		nn.init.xavier_uniform_(self.key_layer.weight)
		nn.init.xavier_uniform_(self.query_layer.weight)
		nn.init.xavier_uniform_(self.attention_value_layer.weight)


		nn.init.xavier_uniform_(self.final_value_layer_1.weight, gain=gain_leaky)
		nn.init.xavier_uniform_(self.final_value_layer_2.weight, gain=gain_leaky)



	def forward(self, states, policies, actions):
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

		Value = F.leaky_relu(self.final_value_layer_1(node_features))
		Value = self.final_value_layer_2(Value)

		return Value, ret_weight


class MultiHeadTransformerCritic(nn.Module):
	def __init__(self, obs_input_dim, obs_output_dim, obs_act_input_dim, obs_act_output_dim, final_input_dim, final_output_dim, num_agents, num_actions, num_heads=2):
		super(MultiHeadTransformerCritic, self).__init__()

		self.name = "MultiHeadTransformerCritic" + str(num_heads)
		
		self.num_agents = num_agents
		self.num_actions = num_actions
		self.num_heads = num_heads
		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		# self.device = "cpu"

		# MULTIHEAD
		self.state_embed_list = []
		self.key_layer_list = []
		self.query_layer_list = []
		self.state_act_pol_embed_list = []
		self.attention_value_layer_list = []
		multi_head_hidden_dim = 128//self.num_heads
		multi_head_obs_output_dim = obs_output_dim//self.num_heads
		multi_head_obs_act_output_dim = obs_act_output_dim//self.num_heads
		for i in range(self.num_heads):
			self.state_embed_list.append(nn.Sequential(nn.Linear(obs_input_dim, multi_head_hidden_dim), nn.LeakyReLU()).to(self.device))
			self.key_layer_list.append(nn.Linear(multi_head_hidden_dim, multi_head_obs_output_dim, bias=False).to(self.device))
			self.query_layer_list.append(nn.Linear(multi_head_hidden_dim, multi_head_obs_output_dim, bias=False).to(self.device))
			self.state_act_pol_embed_list.append(nn.Sequential(nn.Linear(obs_act_input_dim, multi_head_hidden_dim), nn.LeakyReLU()).to(self.device))
			self.attention_value_layer_list.append(nn.Linear(multi_head_hidden_dim, multi_head_obs_act_output_dim, bias=False).to(self.device))
		
		# dimesion of key
		self.d_k_obs_act = multi_head_obs_output_dim

		# NOISE
		self.noise_normal = torch.distributions.Normal(loc=torch.tensor([0.0]), scale=torch.tensor([1.0]))
		self.noise_uniform = torch.rand
		# ********************************************************************************************************

		# ********************************************************************************************************
		# FCN FINAL LAYER TO GET VALUES
		if final_input_dim != multi_head_obs_act_output_dim:
			final_input_dim = multi_head_obs_act_output_dim*self.num_heads

		self.final_value_layer_1 = nn.Linear(final_input_dim, 64, bias=False)
		self.final_value_layer_2 = nn.Linear(64, final_output_dim, bias=False)
		# ********************************************************************************************************	

		self.place_policies = torch.zeros(self.num_agents,self.num_agents,obs_act_input_dim).to(self.device)
		self.place_actions = torch.ones(self.num_agents,self.num_agents,obs_act_input_dim).to(self.device)
		one_hots = torch.ones(obs_act_input_dim)
		zero_hots = torch.zeros(obs_act_input_dim)

		for j in range(self.num_agents):
			self.place_policies[j][j] = one_hots
			self.place_actions[j][j] = zero_hots


		self.reset_parameters()


	def reset_parameters(self):
		"""Reinitialize learnable parameters."""
		gain_leaky = nn.init.calculate_gain('leaky_relu')

		for i in range(self.num_heads):
			nn.init.xavier_uniform_(self.state_embed_list[i][0].weight, gain=gain_leaky)
			nn.init.xavier_uniform_(self.state_act_pol_embed_list[i][0].weight, gain=gain_leaky)

			nn.init.xavier_uniform_(self.key_layer_list[i].weight)
			nn.init.xavier_uniform_(self.query_layer_list[i].weight)
			nn.init.xavier_uniform_(self.attention_value_layer_list[i].weight)


		nn.init.xavier_uniform_(self.final_value_layer_1.weight, gain=gain_leaky)
		nn.init.xavier_uniform_(self.final_value_layer_2.weight, gain=gain_leaky)



	def forward(self, states, policies, actions):
		weights_per_head = []
		node_features_per_head = []

		obs_actions = torch.cat([states,actions],dim=-1)
		obs_policy = torch.cat([states,policies], dim=-1)
		obs_actions = obs_actions.repeat(1,self.num_agents,1).reshape(obs_actions.shape[0],self.num_agents,self.num_agents,-1)
		obs_policy = obs_policy.repeat(1,self.num_agents,1).reshape(obs_policy.shape[0],self.num_agents,self.num_agents,-1)
		obs_actions_policies = self.place_policies*obs_policy + self.place_actions*obs_actions

		for i in range(self.num_heads):
			# EMBED STATES
			states_embed = self.state_embed_list[i](states)
			# KEYS
			key_obs = self.key_layer_list[i](states_embed)
			# QUERIES
			query_obs = self.query_layer_list[i](states_embed)
			# WEIGHT
			weight = F.softmax(torch.matmul(query_obs,key_obs.transpose(1,2))/math.sqrt(self.d_k_obs_act),dim=-1)
			
			weights_per_head.append(weight)

			# EMBED STATE ACTION POLICY
			obs_actions_policies_embed = self.state_act_pol_embed_list[i](obs_actions_policies)
			attention_values = self.attention_value_layer_list[i](obs_actions_policies_embed)
			attention_values = attention_values.repeat(1,self.num_agents,1,1).reshape(attention_values.shape[0],self.num_agents,self.num_agents,self.num_agents,-1)
			
			weight = weight.unsqueeze(-2).repeat(1,1,self.num_agents,1).unsqueeze(-1)
			weighted_attention_values = attention_values*weight
			node_features_per_head.append(torch.sum(weighted_attention_values, dim=-2))

		node_features = torch.cat(node_features_per_head, dim=-1)

		Value = F.leaky_relu(self.final_value_layer_1(node_features))
		Value = self.final_value_layer_2(Value)

		return Value, weights_per_head



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
		self.state_act_pol_embed = nn.Sequential(nn.Linear(obs_act_input_dim, 128), nn.LeakyReLU())
		self.attention_value_layer = nn.Linear(128, obs_act_output_dim, bias=False)
		# dimesion of key
		self.d_k_obs_act = obs_output_dim  

		# NOISE
		self.noise_normal = torch.distributions.Normal(loc=torch.tensor([0.0]), scale=torch.tensor([1.0]))
		self.noise_uniform = torch.rand
		# ********************************************************************************************************

		# ********************************************************************************************************
		# FCN FINAL LAYER TO GET VALUES
		self.final_value_layer_1 = nn.Linear(final_input_dim, 64, bias=False)
		self.final_value_layer_2 = nn.Linear(64, final_output_dim, bias=False)
		# ********************************************************************************************************	

		self.place_policies = torch.zeros(self.num_agents,self.num_agents,obs_act_input_dim).to(self.device)
		self.place_actions = torch.ones(self.num_agents,self.num_agents,obs_act_input_dim).to(self.device)
		one_hots = torch.ones(obs_act_input_dim)
		zero_hots = torch.zeros(obs_act_input_dim)

		for j in range(self.num_agents):
			self.place_policies[j][j] = one_hots
			self.place_actions[j][j] = zero_hots


		self.reset_parameters()


	def reset_parameters(self):
		"""Reinitialize learnable parameters."""
		gain_leaky = nn.init.calculate_gain('leaky_relu')

		nn.init.xavier_uniform_(self.state_embed_preproc[0].weight, gain=gain_leaky)

		nn.init.xavier_uniform_(self.key_layer_preproc.weight)
		nn.init.xavier_uniform_(self.query_layer_preproc.weight)
		nn.init.xavier_uniform_(self.attention_value_layer_preproc.weight)

		nn.init.xavier_uniform_(self.state_embed[0].weight, gain=gain_leaky)
		nn.init.xavier_uniform_(self.state_act_pol_embed[0].weight, gain=gain_leaky)

		nn.init.xavier_uniform_(self.key_layer.weight)
		nn.init.xavier_uniform_(self.query_layer.weight)
		nn.init.xavier_uniform_(self.attention_value_layer.weight)


		nn.init.xavier_uniform_(self.final_value_layer_1.weight, gain=gain_leaky)
		nn.init.xavier_uniform_(self.final_value_layer_2.weight, gain=gain_leaky)



	def forward(self, states, policies, actions):
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

		Value = F.leaky_relu(self.final_value_layer_1(node_features))
		Value = self.final_value_layer_2(Value)

		return Value, weight_preproc, ret_weight


class MultiHeadDualTransformerCritic(nn.Module):
	def __init__(self, obs_input_dim, obs_output_dim, obs_act_input_dim, obs_act_output_dim, final_input_dim, final_output_dim, num_agents, num_actions, num_heads_preproc=2, num_heads_postproc=2):
		super(MultiHeadDualTransformerCritic, self).__init__()

		self.name = "MultiHeadDualTransformerCriticPre" + str(num_heads_preproc) + "Post" + str(num_heads_postproc)
		
		self.num_agents = num_agents
		self.num_actions = num_actions
		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		# self.device = "cpu"
		self.num_heads_preproc = num_heads_preproc
		self.num_heads_postproc = num_heads_postproc

		multi_head_hidden_dim  = 128//self.num_heads_preproc
		multi_head_obs_output_dim = obs_output_dim//self.num_heads_preproc
		self.state_embed_preproc_list = []
		self.key_layer_preproc_list = []
		self.query_layer_preproc_list = []
		self.attention_value_layer_preproc_list = []
		for i in range(self.num_heads_preproc):
			self.state_embed_preproc_list.append(nn.Sequential(nn.Linear(obs_input_dim, multi_head_hidden_dim), nn.LeakyReLU()).to(self.device))
			self.key_layer_preproc_list.append(nn.Linear(multi_head_hidden_dim, multi_head_obs_output_dim, bias=False).to(self.device))
			self.query_layer_preproc_list.append(nn.Linear(multi_head_hidden_dim, multi_head_obs_output_dim, bias=False).to(self.device))
			self.attention_value_layer_preproc_list.append(nn.Linear(multi_head_hidden_dim, multi_head_obs_output_dim, bias=False).to(self.device))
		
		self.d_k_obs = multi_head_obs_output_dim

		multi_head_hidden_dim  = 128//self.num_heads_postproc
		multi_head_obs_output_dim = obs_output_dim//self.num_heads_postproc
		multi_head_obs_act_input_dim = obs_act_input_dim//self.num_heads_postproc
		self.state_embed_list = []
		self.state_act_pol_embed_list = []
		self.key_layer_list = []
		self.query_layer_list = []
		self.attention_value_layer_list = []
		for i in range(self.num_heads_postproc):
			self.state_embed_list.append(nn.Sequential(nn.Linear(obs_output_dim, multi_head_hidden_dim), nn.LeakyReLU()).to(self.device))
			self.key_layer_list.append(nn.Linear(multi_head_hidden_dim, multi_head_obs_output_dim, bias=False).to(self.device))
			self.query_layer_list.append(nn.Linear(multi_head_hidden_dim, multi_head_obs_output_dim, bias=False).to(self.device))
			self.state_act_pol_embed_list.append(nn.Sequential(nn.Linear(obs_act_input_dim, multi_head_hidden_dim), nn.LeakyReLU()).to(self.device))
			self.attention_value_layer_list.append(nn.Linear(multi_head_hidden_dim, multi_head_obs_act_input_dim, bias=False).to(self.device))
		
		# dimesion of key
		self.d_k_obs_act = multi_head_obs_output_dim  

		# NOISE
		self.noise_normal = torch.distributions.Normal(loc=torch.tensor([0.0]), scale=torch.tensor([1.0]))
		self.noise_uniform = torch.rand
		# ********************************************************************************************************

		if final_input_dim != multi_head_obs_act_input_dim*self.num_heads_postproc:
			final_input_dim = multi_head_obs_act_input_dim*self.num_heads_postproc

		# ********************************************************************************************************
		# FCN FINAL LAYER TO GET VALUES
		self.final_value_layer_1 = nn.Linear(final_input_dim, 64, bias=False)
		self.final_value_layer_2 = nn.Linear(64, final_output_dim, bias=False)
		# ********************************************************************************************************	

		self.place_policies = torch.zeros(self.num_agents,self.num_agents,obs_act_input_dim).to(self.device)
		self.place_actions = torch.ones(self.num_agents,self.num_agents,obs_act_input_dim).to(self.device)
		one_hots = torch.ones(obs_act_input_dim)
		zero_hots = torch.zeros(obs_act_input_dim)

		for j in range(self.num_agents):
			self.place_policies[j][j] = one_hots
			self.place_actions[j][j] = zero_hots


		self.reset_parameters()


	def reset_parameters(self):
		"""Reinitialize learnable parameters."""
		gain_leaky = nn.init.calculate_gain('leaky_relu')

		for i in range(self.num_heads_preproc):
			nn.init.xavier_uniform_(self.state_embed_preproc_list[i][0].weight, gain=gain_leaky)

			nn.init.xavier_uniform_(self.key_layer_preproc_list[i].weight)
			nn.init.xavier_uniform_(self.query_layer_preproc_list[i].weight)
			nn.init.xavier_uniform_(self.attention_value_layer_preproc_list[i].weight)

		for i in range(self.num_heads_postproc):
			nn.init.xavier_uniform_(self.state_embed_list[i][0].weight, gain=gain_leaky)
			nn.init.xavier_uniform_(self.state_act_pol_embed_list[i][0].weight, gain=gain_leaky)

			nn.init.xavier_uniform_(self.key_layer_list[i].weight)
			nn.init.xavier_uniform_(self.query_layer_list[i].weight)
			nn.init.xavier_uniform_(self.attention_value_layer_list[i].weight)


		nn.init.xavier_uniform_(self.final_value_layer_1.weight, gain=gain_leaky)
		nn.init.xavier_uniform_(self.final_value_layer_2.weight, gain=gain_leaky)



	def forward(self, states, policies, actions):
		weights_preproc_list = []
		attention_values_preproc_list = []
		for i in range(self.num_heads_preproc):
			# EMBED STATES PREPROC
			states_embed_preproc = self.state_embed_preproc_list[i](states)
			# KEYS
			key_obs_preproc = self.key_layer_list[i](states_embed_preproc)
			# QUERIES
			query_obs_preproc = self.query_layer_list[i](states_embed_preproc)
			# WEIGHT
			weight_preproc = F.softmax(torch.matmul(query_obs_preproc,key_obs_preproc.transpose(1,2))/math.sqrt(self.d_k_obs),dim=-1)
			weights_preproc_list.append(weight_preproc)
			# ATTENTION VALUES
			attention_values_preproc = self.attention_value_layer_preproc_list[i](states_embed_preproc)
			attention_values_preproc_list.append(torch.matmul(weight_preproc, states_embed_preproc))


		attention_values_preproc = torch.cat(attention_values_preproc_list,dim=-1)

		weights_list = []
		attention_values_list = []

		obs_actions = torch.cat([states,actions],dim=-1)
		obs_policy = torch.cat([states,policies], dim=-1)
		obs_actions = obs_actions.repeat(1,self.num_agents,1).reshape(obs_actions.shape[0],self.num_agents,self.num_agents,-1)
		obs_policy = obs_policy.repeat(1,self.num_agents,1).reshape(obs_policy.shape[0],self.num_agents,self.num_agents,-1)
		obs_actions_policies = self.place_policies*obs_policy + self.place_actions*obs_actions

		for i in range(self.num_heads_postproc):
			# EMBED STATES
			states_embed = self.state_embed_list[i](attention_values_preproc)
			# KEYS
			key_obs = self.key_layer_list[i](states_embed)
			# QUERIES
			query_obs = self.query_layer_list[i](states_embed)
			# WEIGHT
			weight = F.softmax(torch.matmul(query_obs,key_obs.transpose(1,2))/math.sqrt(self.d_k_obs_act),dim=-1)
			weights_list.append(weight)

			
			# EMBED STATE ACTION POLICY
			obs_actions_policies_embed = self.state_act_pol_embed_list[i](obs_actions_policies)
			attention_values = self.attention_value_layer_list[i](obs_actions_policies_embed)
			attention_values = attention_values.repeat(1,self.num_agents,1,1).reshape(attention_values.shape[0],self.num_agents,self.num_agents,self.num_agents,-1)
			
			weight = weight.unsqueeze(-2).repeat(1,1,self.num_agents,1).unsqueeze(-1)
			weighted_attention_values = attention_values*weight
			attention_values_list.append(torch.sum(weighted_attention_values, dim=-2))

		node_features = torch.cat(attention_values_list, dim=-1)


		Value = F.leaky_relu(self.final_value_layer_1(node_features))
		Value = self.final_value_layer_2(Value)

		return Value, weights_preproc_list, weights_list


class SemiHardAttnTransformerCritic(nn.Module):
	'''
	https://proceedings.neurips.cc/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf
	'''
	def __init__(self, obs_input_dim, obs_output_dim, obs_act_input_dim, obs_act_output_dim, final_input_dim, final_output_dim, num_agents, num_actions, weight_threshold=None, kth_weight=None):
		super(SemiHardAttnTransformerCritic, self).__init__()

		self.name = "SemiHardAttnTransformerCritic"
		
		self.num_agents = num_agents
		self.num_actions = num_actions
		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		# self.device = "cpu"
		self.weight_threshold = weight_threshold
		self.kth_weight = kth_weight

		self.state_embed = nn.Sequential(nn.Linear(obs_input_dim, 128), nn.LeakyReLU())
		self.key_layer = nn.Linear(128, obs_output_dim, bias=False)
		self.query_layer = nn.Linear(128, obs_output_dim, bias=False)
		self.state_act_pol_embed = nn.Sequential(nn.Linear(obs_act_input_dim, 128), nn.LeakyReLU())
		self.attention_value_layer = nn.Linear(128, obs_act_output_dim, bias=False)
		# dimesion of key
		self.d_k_obs_act = obs_output_dim  

		# NOISE
		self.noise_normal = torch.distributions.Normal(loc=torch.tensor([0.0]), scale=torch.tensor([1.0]))
		self.noise_uniform = torch.rand
		# ********************************************************************************************************

		# ********************************************************************************************************
		# FCN FINAL LAYER TO GET VALUES
		self.final_value_layer_1 = nn.Linear(final_input_dim, 64, bias=False)
		self.final_value_layer_2 = nn.Linear(64, final_output_dim, bias=False)
		# ********************************************************************************************************	

		self.place_policies = torch.zeros(self.num_agents,self.num_agents,obs_act_input_dim).to(self.device)
		self.place_actions = torch.ones(self.num_agents,self.num_agents,obs_act_input_dim).to(self.device)
		one_hots = torch.ones(obs_act_input_dim)
		zero_hots = torch.zeros(obs_act_input_dim)

		for j in range(self.num_agents):
			self.place_policies[j][j] = one_hots
			self.place_actions[j][j] = zero_hots


		self.reset_parameters()


	def reset_parameters(self):
		"""Reinitialize learnable parameters."""
		gain_leaky = nn.init.calculate_gain('leaky_relu')

		nn.init.xavier_uniform_(self.state_embed[0].weight, gain=gain_leaky)
		nn.init.xavier_uniform_(self.state_act_pol_embed[0].weight, gain=gain_leaky)

		nn.init.xavier_uniform_(self.key_layer.weight)
		nn.init.xavier_uniform_(self.query_layer.weight)
		nn.init.xavier_uniform_(self.attention_value_layer.weight)


		nn.init.xavier_uniform_(self.final_value_layer_1.weight, gain=gain_leaky)
		nn.init.xavier_uniform_(self.final_value_layer_2.weight, gain=gain_leaky)



	def forward(self, states, policies, actions):
		# EMBED STATES
		states_embed = self.state_embed(states)
		# KEYS
		key_obs = self.key_layer(states_embed)
		# QUERIES
		query_obs = self.query_layer(states_embed)
		# WEIGHT
		weight = F.softmax(torch.matmul(query_obs,key_obs.transpose(1,2))/math.sqrt(self.d_k_obs_act),dim=-1)

		if self.kth_weight is not None:
			weight_sub, indices = torch.topk(weight,k=self.kth_weight,dim=-1)
			weight_sub = weight_sub[:,:,-1].unsqueeze(-1)
			weight = F.relu(weight-weight_sub)
			weight = F.softmax(weight,dim=-1)
		elif self.weight_threshold is not None:
			weight = F.relu(weight - self.weight_threshold)
			weight = F.softmax(weight,dim=-1)

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

		Value = F.leaky_relu(self.final_value_layer_1(node_features))
		Value = self.final_value_layer_2(Value)

		return Value, ret_weight


class MultiHeadSemiHardAttnTransformerCritic(nn.Module):
	'''
	https://proceedings.neurips.cc/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf
	'''
	def __init__(self, obs_input_dim, obs_output_dim, obs_act_input_dim, obs_act_output_dim, final_input_dim, final_output_dim, num_agents, num_actions, weight_threshold=None, kth_weight=None, num_heads=2):
		super(MultiHeadSemiHardAttnTransformerCritic, self).__init__()

		self.name = "MultiHeadSemiHardAttnTransformerCritic" + str(num_heads)
		
		self.num_agents = num_agents
		self.num_actions = num_actions
		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		# self.device = "cpu"
		self.weight_threshold = weight_threshold
		self.kth_weight = kth_weight
		self.num_heads = num_heads


		self.state_embed_list = []
		self.key_layer_list = []
		self.query_layer_list = []
		self.state_act_pol_embed_list = []
		self.attention_value_layer_list = []
		multi_head_hidden_dim = 128//self.num_heads
		multi_head_obs_output_dim = obs_output_dim//self.num_heads
		multi_head_obs_act_output_dim = obs_act_output_dim//self.num_heads
		for i in range(self.num_heads):
			self.state_embed_list.append(nn.Sequential(nn.Linear(obs_input_dim, multi_head_hidden_dim), nn.LeakyReLU()).to(self.device))
			self.key_layer_list.append(nn.Linear(multi_head_hidden_dim, multi_head_obs_output_dim, bias=False).to(self.device))
			self.query_layer_list.append(nn.Linear(multi_head_hidden_dim, multi_head_obs_output_dim, bias=False).to(self.device))
			self.state_act_pol_embed_list.append(nn.Sequential(nn.Linear(obs_act_input_dim, multi_head_hidden_dim), nn.LeakyReLU()).to(self.device))
			self.attention_value_layer_list.append(nn.Linear(multi_head_hidden_dim, multi_head_obs_act_output_dim, bias=False).to(self.device))
		# dimesion of key
		self.d_k_obs_act = multi_head_obs_output_dim  

		# NOISE
		self.noise_normal = torch.distributions.Normal(loc=torch.tensor([0.0]), scale=torch.tensor([1.0]))
		self.noise_uniform = torch.rand
		# ********************************************************************************************************

		# ********************************************************************************************************
		# FCN FINAL LAYER TO GET VALUES
		self.final_value_layer_1 = nn.Linear(final_input_dim, 64, bias=False)
		self.final_value_layer_2 = nn.Linear(64, final_output_dim, bias=False)
		# ********************************************************************************************************	

		self.place_policies = torch.zeros(self.num_agents,self.num_agents,obs_act_input_dim).to(self.device)
		self.place_actions = torch.ones(self.num_agents,self.num_agents,obs_act_input_dim).to(self.device)
		one_hots = torch.ones(obs_act_input_dim)
		zero_hots = torch.zeros(obs_act_input_dim)

		for j in range(self.num_agents):
			self.place_policies[j][j] = one_hots
			self.place_actions[j][j] = zero_hots


		self.reset_parameters()


	def reset_parameters(self):
		"""Reinitialize learnable parameters."""
		gain_leaky = nn.init.calculate_gain('leaky_relu')

		for i in range(self.num_heads):
			nn.init.xavier_uniform_(self.state_embed_list[i][0].weight, gain=gain_leaky)
			nn.init.xavier_uniform_(self.state_act_pol_embed_list[i][0].weight, gain=gain_leaky)

			nn.init.xavier_uniform_(self.key_layer_list[i].weight)
			nn.init.xavier_uniform_(self.query_layer_list[i].weight)
			nn.init.xavier_uniform_(self.attention_value_layer_list[i].weight)


		nn.init.xavier_uniform_(self.final_value_layer_1.weight, gain=gain_leaky)
		nn.init.xavier_uniform_(self.final_value_layer_2.weight, gain=gain_leaky)



	def forward(self, states, policies, actions):

		obs_actions = torch.cat([states,actions],dim=-1)
		obs_policy = torch.cat([states,policies], dim=-1)
		obs_actions = obs_actions.repeat(1,self.num_agents,1).reshape(obs_actions.shape[0],self.num_agents,self.num_agents,-1)
		obs_policy = obs_policy.repeat(1,self.num_agents,1).reshape(obs_policy.shape[0],self.num_agents,self.num_agents,-1)
		obs_actions_policies = self.place_policies*obs_policy + self.place_actions*obs_actions

		attention_values_list = []
		weights_list = []

		for i in range(self.num_heads):
			# EMBED STATES
			states_embed = self.state_embed_list[i](states)
			# KEYS
			key_obs = self.key_layer_list[i](states_embed)
			# QUERIES
			query_obs = self.query_layer_list[i](states_embed)
			# WEIGHT
			weight = F.softmax(torch.matmul(query_obs,key_obs.transpose(1,2))/math.sqrt(self.d_k_obs_act),dim=-1)

			if self.kth_weight is not None:
				weight_sub, indices = torch.topk(weight,k=self.kth_weight,dim=-1)
				weight_sub = weight_sub[:,:,-1].unsqueeze(-1)
				weight = F.relu(weight-weight_sub)
				weight = F.softmax(weight,dim=-1)
			elif self.weight_threshold is not None:
				weight = F.relu(weight - self.weight_threshold)
				weight = F.softmax(weight,dim=-1)

			weights_list.append(weight)
			ret_weight = weight

			
			# EMBED STATE ACTION POLICY
			obs_actions_policies_embed = self.state_act_pol_embed_list[i](obs_actions_policies)
			attention_values = self.attention_value_layer_list[i](obs_actions_policies_embed)
			attention_values = attention_values.repeat(1,self.num_agents,1,1).reshape(attention_values.shape[0],self.num_agents,self.num_agents,self.num_agents,-1)
			
			weight = weight.unsqueeze(-2).repeat(1,1,self.num_agents,1).unsqueeze(-1)
			weighted_attention_values = attention_values*weight
			attention_values_list.append(torch.sum(weighted_attention_values, dim=-2))

		node_features = torch.cat(attention_values_list, dim=-1)

		Value = F.leaky_relu(self.final_value_layer_1(node_features))
		Value = self.final_value_layer_2(Value)

		return Value, weights_list


'''
GAT Arch
'''
class GATCritic(nn.Module):
	'''
	https://arxiv.org/pdf/1710.10903.pdf
	'''
	def __init__(self, obs_input_dim, obs_output_dim, obs_act_input_dim, obs_act_output_dim, final_input_dim, final_output_dim, num_agents, num_actions):
		super(GATCritic, self).__init__()

		self.name = "GATCritic"
		
		self.num_agents = num_agents
		self.num_actions = num_actions
		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		# self.device = "cpu"

		self.state_embed = nn.Linear(obs_input_dim, obs_output_dim)
		self.state_act_pol_embed = nn.Sequential(nn.Linear(obs_act_input_dim, obs_act_output_dim), nn.LeakyReLU())
		self.attention_network = nn.Sequential(nn.Linear(obs_output_dim*2, 1), nn.LeakyReLU())

		# NOISE
		self.noise_normal = torch.distributions.Normal(loc=torch.tensor([0.0]), scale=torch.tensor([1.0]))
		self.noise_uniform = torch.rand
		# ********************************************************************************************************

		# ********************************************************************************************************
		# FCN FINAL LAYER TO GET VALUES
		self.final_value_layer_1 = nn.Linear(final_input_dim, 64, bias=False)
		self.final_value_layer_2 = nn.Linear(64, final_output_dim, bias=False)
		# ********************************************************************************************************	

		self.place_policies = torch.zeros(self.num_agents,self.num_agents,obs_act_input_dim).to(self.device)
		self.place_actions = torch.ones(self.num_agents,self.num_agents,obs_act_input_dim).to(self.device)
		one_hots = torch.ones(obs_act_input_dim)
		zero_hots = torch.zeros(obs_act_input_dim)

		for j in range(self.num_agents):
			self.place_policies[j][j] = one_hots
			self.place_actions[j][j] = zero_hots


		self.reset_parameters()


	def reset_parameters(self):
		"""Reinitialize learnable parameters."""
		gain_leaky = nn.init.calculate_gain('leaky_relu')

		nn.init.xavier_uniform_(self.state_embed.weight)
		nn.init.xavier_uniform_(self.state_act_pol_embed[0].weight, gain=gain_leaky)

		nn.init.xavier_uniform_(self.attention_network[0].weight, gain=gain_leaky)

		nn.init.xavier_uniform_(self.final_value_layer_1.weight, gain=gain_leaky)
		nn.init.xavier_uniform_(self.final_value_layer_2.weight, gain=gain_leaky)


	def forward(self, states, policies, actions):
		# EMBED STATES
		states_embed = self.state_embed(states)

		source_state_embed = states_embed.unsqueeze(2).repeat(1,1,self.num_agents,1)
		destination_state_embed = states_embed.unsqueeze(1).repeat(1,self.num_agents,1,1)
		# WEIGHT
		weight_input = torch.cat([source_state_embed, destination_state_embed], dim=-1)
		weight = F.softmax(self.attention_network(weight_input).squeeze(-1),dim=-1)
		ret_weight = weight

		obs_actions = torch.cat([states,actions],dim=-1)
		obs_policy = torch.cat([states,policies], dim=-1)
		obs_actions = obs_actions.repeat(1,self.num_agents,1).reshape(obs_actions.shape[0],self.num_agents,self.num_agents,-1)
		obs_policy = obs_policy.repeat(1,self.num_agents,1).reshape(obs_policy.shape[0],self.num_agents,self.num_agents,-1)
		obs_actions_policies = self.place_policies*obs_policy + self.place_actions*obs_actions
		# EMBED STATE ACTION POLICY
		obs_actions_policies_embed = self.state_act_pol_embed(obs_actions_policies)
		obs_actions_policies_embed = obs_actions_policies_embed.repeat(1,self.num_agents,1,1).reshape(obs_actions_policies_embed.shape[0],self.num_agents,self.num_agents,self.num_agents,-1)
		weight = weight.unsqueeze(-2).repeat(1,1,self.num_agents,1).unsqueeze(-1)
		weighted_attention_values = obs_actions_policies_embed*weight
		node_features = torch.sum(weighted_attention_values, dim=-2)

		Value = F.leaky_relu(self.final_value_layer_1(node_features))
		Value = self.final_value_layer_2(Value)

		return Value, ret_weight



class MultiHeadGATCritic(nn.Module):
	'''
	https://arxiv.org/pdf/1710.10903.pdf
	'''
	def __init__(self, obs_input_dim, obs_output_dim, obs_act_input_dim, obs_act_output_dim, final_input_dim, final_output_dim, num_agents, num_actions, num_heads=2):
		super(MultiHeadGATCritic, self).__init__()

		self.name = "MultiHeadGATCritic" + str(num_heads)
		
		self.num_agents = num_agents
		self.num_actions = num_actions
		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		# self.device = "cpu"
		self.num_heads = num_heads

		self.state_embed_list = []
		self.state_act_pol_embed_list = []
		self.attention_network_list = []
		multi_head_obs_output_dim = obs_output_dim//self.num_heads
		multi_head_obs_act_output_dim = obs_act_output_dim//self.num_heads
		for i in range(self.num_heads):
			self.state_embed_list.append(nn.Linear(obs_input_dim, multi_head_obs_output_dim).to(self.device))
			self.state_act_pol_embed_list.append(nn.Sequential(nn.Linear(obs_act_input_dim, multi_head_obs_act_output_dim), nn.LeakyReLU()).to(self.device))
			self.attention_network_list.append(nn.Sequential(nn.Linear(multi_head_obs_output_dim*2, 1), nn.LeakyReLU()).to(self.device))

		# NOISE
		self.noise_normal = torch.distributions.Normal(loc=torch.tensor([0.0]), scale=torch.tensor([1.0]))
		self.noise_uniform = torch.rand
		# ********************************************************************************************************

		# ********************************************************************************************************
		# FCN FINAL LAYER TO GET VALUES
		if final_input_dim != multi_head_obs_act_output_dim*self.num_heads:
			final_input_dim = multi_head_obs_act_output_dim*self.num_heads

		self.final_value_layer_1 = nn.Linear(final_input_dim, 64, bias=False)
		self.final_value_layer_2 = nn.Linear(64, final_output_dim, bias=False)
		# ********************************************************************************************************	

		self.place_policies = torch.zeros(self.num_agents,self.num_agents,obs_act_input_dim).to(self.device)
		self.place_actions = torch.ones(self.num_agents,self.num_agents,obs_act_input_dim).to(self.device)
		one_hots = torch.ones(obs_act_input_dim)
		zero_hots = torch.zeros(obs_act_input_dim)

		for j in range(self.num_agents):
			self.place_policies[j][j] = one_hots
			self.place_actions[j][j] = zero_hots


		self.reset_parameters()


	def reset_parameters(self):
		"""Reinitialize learnable parameters."""
		gain_leaky = nn.init.calculate_gain('leaky_relu')

		for i in range(self.num_heads):
			nn.init.xavier_uniform_(self.state_embed_list[i].weight)
			nn.init.xavier_uniform_(self.state_act_pol_embed_list[i][0].weight, gain=gain_leaky)

			nn.init.xavier_uniform_(self.attention_network_list[i][0].weight, gain=gain_leaky)

		nn.init.xavier_uniform_(self.final_value_layer_1.weight, gain=gain_leaky)
		nn.init.xavier_uniform_(self.final_value_layer_2.weight, gain=gain_leaky)


	def forward(self, states, policies, actions):

		obs_actions = torch.cat([states,actions],dim=-1)
		obs_policy = torch.cat([states,policies], dim=-1)
		obs_actions = obs_actions.repeat(1,self.num_agents,1).reshape(obs_actions.shape[0],self.num_agents,self.num_agents,-1)
		obs_policy = obs_policy.repeat(1,self.num_agents,1).reshape(obs_policy.shape[0],self.num_agents,self.num_agents,-1)
		obs_actions_policies = self.place_policies*obs_policy + self.place_actions*obs_actions

		weights_list = []
		attention_values_list = []

		for i in range(self.num_heads):
			# EMBED STATES
			states_embed = self.state_embed_list[i](states)

			source_state_embed = states_embed.unsqueeze(2).repeat(1,1,self.num_agents,1)
			destination_state_embed = states_embed.unsqueeze(1).repeat(1,self.num_agents,1,1)
			# WEIGHT
			weight_input = torch.cat([source_state_embed, destination_state_embed], dim=-1)
			weight = F.softmax(self.attention_network_list[i](weight_input).squeeze(-1),dim=-1)
			weights_list.append(weight)

			# EMBED STATE ACTION POLICY
			obs_actions_policies_embed = self.state_act_pol_embed_list[i](obs_actions_policies)
			obs_actions_policies_embed = obs_actions_policies_embed.repeat(1,self.num_agents,1,1).reshape(obs_actions_policies_embed.shape[0],self.num_agents,self.num_agents,self.num_agents,-1)
			weight = weight.unsqueeze(-2).repeat(1,1,self.num_agents,1).unsqueeze(-1)
			weighted_attention_values = obs_actions_policies_embed*weight
			attention_values_list.append(torch.sum(weighted_attention_values, dim=-2))

		node_features = torch.cat(attention_values_list, dim=-1)

		Value = F.leaky_relu(self.final_value_layer_1(node_features))
		Value = self.final_value_layer_2(Value)

		return Value, weights_list



class SemiHardGATCritic(nn.Module):
	'''
	https://arxiv.org/pdf/1710.10903.pdf
	'''
	def __init__(self, obs_input_dim, obs_output_dim, obs_act_input_dim, obs_act_output_dim, final_input_dim, final_output_dim, num_agents, num_actions, weight_threshold=None, kth_weight=None):
		super(SemiHardGATCritic, self).__init__()

		self.name = "SemiHardGATCritic"
		
		self.num_agents = num_agents
		self.num_actions = num_actions
		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		# self.device = "cpu"
		self.weight_threshold = weight_threshold
		self.kth_weight = kth_weight

		self.state_embed = nn.Linear(obs_input_dim, obs_output_dim)
		self.state_act_pol_embed = nn.Sequential(nn.Linear(obs_act_input_dim, obs_act_output_dim), nn.LeakyReLU())
		self.attention_network = nn.Sequential(nn.Linear(obs_output_dim*2, 1), nn.LeakyReLU())

		# NOISE
		self.noise_normal = torch.distributions.Normal(loc=torch.tensor([0.0]), scale=torch.tensor([1.0]))
		self.noise_uniform = torch.rand
		# ********************************************************************************************************

		# ********************************************************************************************************
		# FCN FINAL LAYER TO GET VALUES
		self.final_value_layer_1 = nn.Linear(final_input_dim, 64, bias=False)
		self.final_value_layer_2 = nn.Linear(64, final_output_dim, bias=False)
		# ********************************************************************************************************	

		self.place_policies = torch.zeros(self.num_agents,self.num_agents,obs_act_input_dim).to(self.device)
		self.place_actions = torch.ones(self.num_agents,self.num_agents,obs_act_input_dim).to(self.device)
		one_hots = torch.ones(obs_act_input_dim)
		zero_hots = torch.zeros(obs_act_input_dim)

		for j in range(self.num_agents):
			self.place_policies[j][j] = one_hots
			self.place_actions[j][j] = zero_hots


		self.reset_parameters()


	def reset_parameters(self):
		"""Reinitialize learnable parameters."""
		gain_leaky = nn.init.calculate_gain('leaky_relu')

		nn.init.xavier_uniform_(self.state_embed.weight)
		nn.init.xavier_uniform_(self.state_act_pol_embed[0].weight, gain=gain_leaky)
		nn.init.xavier_uniform_(self.attention_network[0].weight, gain=gain_leaky)

		nn.init.xavier_uniform_(self.final_value_layer_1.weight, gain=gain_leaky)
		nn.init.xavier_uniform_(self.final_value_layer_2.weight, gain=gain_leaky)


	def forward(self, states, policies, actions):
		# EMBED STATES
		states_embed = self.state_embed(states)

		source_state_embed = states_embed.unsqueeze(2).repeat(1,1,self.num_agents,1)
		destination_state_embed = states_embed.unsqueeze(1).repeat(1,self.num_agents,1,1)
		# WEIGHT
		weight_input = torch.cat([source_state_embed, destination_state_embed], dim=-1)
		weight = F.softmax(self.attention_network(weight_input).squeeze(-1),dim=-1)

		if self.kth_weight is not None:
			weight_sub, indices = torch.topk(weight,k=self.kth_weight,dim=-1)
			weight_sub = weight_sub[:,:,-1].unsqueeze(-1)
			weight = F.relu(weight-weight_sub)
			weight = F.softmax(weight,dim=-1)
		elif self.weight_threshold is not None:
			weight = F.relu(weight - self.weight_threshold)
			weight = F.softmax(weight,dim=-1)
		
		ret_weight = weight

		obs_actions = torch.cat([states,actions],dim=-1)
		obs_policy = torch.cat([states,policies], dim=-1)
		obs_actions = obs_actions.repeat(1,self.num_agents,1).reshape(obs_actions.shape[0],self.num_agents,self.num_agents,-1)
		obs_policy = obs_policy.repeat(1,self.num_agents,1).reshape(obs_policy.shape[0],self.num_agents,self.num_agents,-1)
		obs_actions_policies = self.place_policies*obs_policy + self.place_actions*obs_actions
		# EMBED STATE ACTION POLICY
		obs_actions_policies_embed = self.state_act_pol_embed(obs_actions_policies)
		obs_actions_policies_embed = obs_actions_policies_embed.repeat(1,self.num_agents,1,1).reshape(obs_actions_policies_embed.shape[0],self.num_agents,self.num_agents,self.num_agents,-1)
		weight = weight.unsqueeze(-2).repeat(1,1,self.num_agents,1).unsqueeze(-1)
		weighted_attention_values = obs_actions_policies_embed*weight
		node_features = torch.sum(weighted_attention_values, dim=-2)

		Value = F.leaky_relu(self.final_value_layer_1(node_features))
		Value = self.final_value_layer_2(Value)

		return Value, ret_weight



class SemiHardMultiHeadGATCritic(nn.Module):
	'''
	https://arxiv.org/pdf/1710.10903.pdf
	'''
	def __init__(self, obs_input_dim, obs_output_dim, obs_act_input_dim, obs_act_output_dim, final_input_dim, final_output_dim, num_agents, num_actions, weight_threshold=None, kth_weight=None, num_heads=2):
		super(SemiHardMultiHeadGATCritic, self).__init__()

		self.name = "SemiHardMultiHeadGATCritic" + str(num_heads)
		
		self.num_agents = num_agents
		self.num_actions = num_actions
		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		# self.device = "cpu"
		self.num_heads = num_heads
		self.weight_threshold = weight_threshold
		self.kth_weight = kth_weight

		self.state_embed_list = []
		self.state_act_pol_embed_list = []
		self.attention_network_list = []
		multi_head_obs_output_dim = obs_output_dim//self.num_heads
		multi_head_obs_act_output_dim = obs_act_output_dim//self.num_heads
		for i in range(self.num_heads):
			self.state_embed_list.append(nn.Linear(obs_input_dim, multi_head_obs_output_dim).to(self.device))
			self.state_act_pol_embed_list.append(nn.Sequential(nn.Linear(obs_act_input_dim, multi_head_obs_act_output_dim), nn.LeakyReLU()).to(self.device))
			self.attention_network_list.append(nn.Sequential(nn.Linear(multi_head_obs_output_dim*2, 1), nn.LeakyReLU()).to(self.device))

		# NOISE
		self.noise_normal = torch.distributions.Normal(loc=torch.tensor([0.0]), scale=torch.tensor([1.0]))
		self.noise_uniform = torch.rand
		# ********************************************************************************************************

		# ********************************************************************************************************
		# FCN FINAL LAYER TO GET VALUES
		if final_input_dim != multi_head_obs_act_output_dim*self.num_heads:
			final_input_dim = multi_head_obs_act_output_dim*self.num_heads

		self.final_value_layer_1 = nn.Linear(final_input_dim, 64, bias=False)
		self.final_value_layer_2 = nn.Linear(64, final_output_dim, bias=False)
		# ********************************************************************************************************	

		self.place_policies = torch.zeros(self.num_agents,self.num_agents,obs_act_input_dim).to(self.device)
		self.place_actions = torch.ones(self.num_agents,self.num_agents,obs_act_input_dim).to(self.device)
		one_hots = torch.ones(obs_act_input_dim)
		zero_hots = torch.zeros(obs_act_input_dim)

		for j in range(self.num_agents):
			self.place_policies[j][j] = one_hots
			self.place_actions[j][j] = zero_hots


		self.reset_parameters()


	def reset_parameters(self):
		"""Reinitialize learnable parameters."""
		gain_leaky = nn.init.calculate_gain('leaky_relu')

		for i in range(self.num_heads):
			nn.init.xavier_uniform_(self.state_embed_list[i].weight)
			nn.init.xavier_uniform_(self.state_act_pol_embed_list[i][0].weight, gain=gain_leaky)

			nn.init.xavier_uniform_(self.attention_network_list[i][0].weight, gain=gain_leaky)

		nn.init.xavier_uniform_(self.final_value_layer_1.weight, gain=gain_leaky)
		nn.init.xavier_uniform_(self.final_value_layer_2.weight, gain=gain_leaky)


	def forward(self, states, policies, actions):

		obs_actions = torch.cat([states,actions],dim=-1)
		obs_policy = torch.cat([states,policies], dim=-1)
		obs_actions = obs_actions.repeat(1,self.num_agents,1).reshape(obs_actions.shape[0],self.num_agents,self.num_agents,-1)
		obs_policy = obs_policy.repeat(1,self.num_agents,1).reshape(obs_policy.shape[0],self.num_agents,self.num_agents,-1)
		obs_actions_policies = self.place_policies*obs_policy + self.place_actions*obs_actions

		weights_list = []
		attention_values_list = []

		for i in range(self.num_heads):
			# EMBED STATES
			states_embed = self.state_embed_list[i](states)

			source_state_embed = states_embed.unsqueeze(2).repeat(1,1,self.num_agents,1)
			destination_state_embed = states_embed.unsqueeze(1).repeat(1,self.num_agents,1,1)
			# WEIGHT
			weight_input = torch.cat([source_state_embed, destination_state_embed], dim=-1)
			weight = F.softmax(self.attention_network_list[i](weight_input).squeeze(-1),dim=-1)

			if self.kth_weight is not None:
				weight_sub, indices = torch.topk(weight,k=self.kth_weight,dim=-1)
				weight_sub = weight_sub[:,:,-1].unsqueeze(-1)
				weight = F.relu(weight-weight_sub)
				weight = F.softmax(weight,dim=-1)
			elif self.weight_threshold is not None:
				weight = F.relu(weight - self.weight_threshold)
				weight = F.softmax(weight,dim=-1)

			weights_list.append(weight)

			# EMBED STATE ACTION POLICY
			obs_actions_policies_embed = self.state_act_pol_embed_list[i](obs_actions_policies)
			obs_actions_policies_embed = obs_actions_policies_embed.repeat(1,self.num_agents,1,1).reshape(obs_actions_policies_embed.shape[0],self.num_agents,self.num_agents,self.num_agents,-1)
			weight = weight.unsqueeze(-2).repeat(1,1,self.num_agents,1).unsqueeze(-1)
			weighted_attention_values = obs_actions_policies_embed*weight
			attention_values_list.append(torch.sum(weighted_attention_values, dim=-2))

		node_features = torch.cat(attention_values_list, dim=-1)

		Value = F.leaky_relu(self.final_value_layer_1(node_features))
		Value = self.final_value_layer_2(Value)

		return Value, weights_list


'''
GATv2 --> Improved version of GAT for dynamic weight assignment
'''

class GATV2Critic(nn.Module):
	'''
	https://arxiv.org/pdf/2105.14491.pdf
	'''
	def __init__(self, obs_input_dim, obs_output_dim, obs_act_input_dim, obs_act_output_dim, final_input_dim, final_output_dim, num_agents, num_actions):
		super(GATV2Critic, self).__init__()

		self.name = "GATV2Critic"
		
		self.num_agents = num_agents
		self.num_actions = num_actions
		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		# self.device = "cpu"

		self.state_embed = nn.Sequential(nn.Linear(obs_input_dim*2, obs_output_dim), nn.LeakyReLU())
		self.state_act_pol_embed = nn.Sequential(nn.Linear(obs_act_input_dim, obs_act_output_dim), nn.LeakyReLU())
		self.attention_network = nn.Linear(obs_output_dim, 1)

		# NOISE
		self.noise_normal = torch.distributions.Normal(loc=torch.tensor([0.0]), scale=torch.tensor([1.0]))
		self.noise_uniform = torch.rand
		# ********************************************************************************************************

		# ********************************************************************************************************
		# FCN FINAL LAYER TO GET VALUES
		self.final_value_layer_1 = nn.Linear(final_input_dim, 64, bias=False)
		self.final_value_layer_2 = nn.Linear(64, final_output_dim, bias=False)
		# ********************************************************************************************************	

		self.place_policies = torch.zeros(self.num_agents,self.num_agents,obs_act_input_dim).to(self.device)
		self.place_actions = torch.ones(self.num_agents,self.num_agents,obs_act_input_dim).to(self.device)
		one_hots = torch.ones(obs_act_input_dim)
		zero_hots = torch.zeros(obs_act_input_dim)

		for j in range(self.num_agents):
			self.place_policies[j][j] = one_hots
			self.place_actions[j][j] = zero_hots


		self.reset_parameters()


	def reset_parameters(self):
		"""Reinitialize learnable parameters."""
		gain_leaky = nn.init.calculate_gain('leaky_relu')

		nn.init.xavier_uniform_(self.state_embed[0].weight, gain=gain_leaky)
		nn.init.xavier_uniform_(self.state_act_pol_embed[0].weight, gain=gain_leaky)

		nn.init.xavier_uniform_(self.attention_network.weight)

		nn.init.xavier_uniform_(self.final_value_layer_1.weight, gain=gain_leaky)
		nn.init.xavier_uniform_(self.final_value_layer_2.weight, gain=gain_leaky)


	def forward(self, states, policies, actions):

		source_state = states.unsqueeze(2).repeat(1,1,self.num_agents,1)
		destination_state = states.unsqueeze(1).repeat(1,self.num_agents,1,1)
		# WEIGHT
		state_embed_input = torch.cat([source_state, destination_state], dim=-1)
		states_embed = self.state_embed(state_embed_input)
		weight = F.softmax(self.attention_network(states_embed).squeeze(-1),dim=-1)
		ret_weight = weight

		obs_actions = torch.cat([states,actions],dim=-1)
		obs_policy = torch.cat([states,policies], dim=-1)
		obs_actions = obs_actions.repeat(1,self.num_agents,1).reshape(obs_actions.shape[0],self.num_agents,self.num_agents,-1)
		obs_policy = obs_policy.repeat(1,self.num_agents,1).reshape(obs_policy.shape[0],self.num_agents,self.num_agents,-1)
		obs_actions_policies = self.place_policies*obs_policy + self.place_actions*obs_actions
		# EMBED STATE ACTION POLICY
		obs_actions_policies_embed = self.state_act_pol_embed(obs_actions_policies)
		obs_actions_policies_embed = obs_actions_policies_embed.repeat(1,self.num_agents,1,1).reshape(obs_actions_policies_embed.shape[0],self.num_agents,self.num_agents,self.num_agents,-1)
		weight = weight.unsqueeze(-2).repeat(1,1,self.num_agents,1).unsqueeze(-1)
		weighted_attention_values = obs_actions_policies_embed*weight
		node_features = torch.sum(weighted_attention_values, dim=-2)

		Value = F.leaky_relu(self.final_value_layer_1(node_features))
		Value = self.final_value_layer_2(Value)

		return Value, ret_weight



class MultiHeadGATV2Critic(nn.Module):
	'''
	https://arxiv.org/pdf/2105.14491.pdf
	'''
	def __init__(self, obs_input_dim, obs_output_dim, obs_act_input_dim, obs_act_output_dim, final_input_dim, final_output_dim, num_agents, num_actions, num_heads=2):
		super(MultiHeadGATV2Critic, self).__init__()

		self.name = "MultiHeadGATV2Critic" + str(num_heads)
		
		self.num_agents = num_agents
		self.num_actions = num_actions
		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		# self.device = "cpu"
		self.num_heads = num_heads

		self.state_embed_list = []
		self.state_act_pol_embed_list = []
		self.attention_network_list = []

		multi_head_obs_output_dim = obs_output_dim//self.num_heads
		multi_head_obs_act_output_dim = obs_act_output_dim//self.num_heads

		for i in range(self.num_heads):
			self.state_embed_list.append(nn.Sequential(nn.Linear(obs_input_dim*2, multi_head_obs_output_dim), nn.LeakyReLU()).to(self.device))
			self.state_act_pol_embed_list.append(nn.Sequential(nn.Linear(obs_act_input_dim, multi_head_obs_act_output_dim), nn.LeakyReLU()).to(self.device))
			self.attention_network_list.append(nn.Linear(multi_head_obs_output_dim, 1).to(self.device))

		# NOISE
		self.noise_normal = torch.distributions.Normal(loc=torch.tensor([0.0]), scale=torch.tensor([1.0]))
		self.noise_uniform = torch.rand
		# ********************************************************************************************************

		# ********************************************************************************************************
		# FCN FINAL LAYER TO GET VALUES
		if final_input_dim != multi_head_obs_act_output_dim*self.num_heads:
			final_input_dim = multi_head_obs_act_output_dim*self.num_heads
		self.final_value_layer_1 = nn.Linear(final_input_dim, 64, bias=False)
		self.final_value_layer_2 = nn.Linear(64, final_output_dim, bias=False)
		# ********************************************************************************************************	

		self.place_policies = torch.zeros(self.num_agents,self.num_agents,obs_act_input_dim).to(self.device)
		self.place_actions = torch.ones(self.num_agents,self.num_agents,obs_act_input_dim).to(self.device)
		one_hots = torch.ones(obs_act_input_dim)
		zero_hots = torch.zeros(obs_act_input_dim)

		for j in range(self.num_agents):
			self.place_policies[j][j] = one_hots
			self.place_actions[j][j] = zero_hots


		self.reset_parameters()


	def reset_parameters(self):
		"""Reinitialize learnable parameters."""
		gain_leaky = nn.init.calculate_gain('leaky_relu')

		for i in range(self.num_heads):
			nn.init.xavier_uniform_(self.state_embed_list[i][0].weight, gain=gain_leaky)
			nn.init.xavier_uniform_(self.state_act_pol_embed_list[i][0].weight, gain=gain_leaky)
			nn.init.xavier_uniform_(self.attention_network_list[i].weight)

		nn.init.xavier_uniform_(self.final_value_layer_1.weight, gain=gain_leaky)
		nn.init.xavier_uniform_(self.final_value_layer_2.weight, gain=gain_leaky)


	def forward(self, states, policies, actions):

		source_state = states.unsqueeze(2).repeat(1,1,self.num_agents,1)
		destination_state = states.unsqueeze(1).repeat(1,self.num_agents,1,1)
		# WEIGHT
		state_embed_input = torch.cat([source_state, destination_state], dim=-1)

		weights_list = []
		attention_values_list = []
		for i in range(self.num_heads):
			states_embed = self.state_embed_list[i](state_embed_input)
			weight = F.softmax(self.attention_network_list[i](states_embed).squeeze(-1),dim=-1)
			weights_list.append(weight)

			obs_actions = torch.cat([states,actions],dim=-1)
			obs_policy = torch.cat([states,policies], dim=-1)
			obs_actions = obs_actions.repeat(1,self.num_agents,1).reshape(obs_actions.shape[0],self.num_agents,self.num_agents,-1)
			obs_policy = obs_policy.repeat(1,self.num_agents,1).reshape(obs_policy.shape[0],self.num_agents,self.num_agents,-1)
			obs_actions_policies = self.place_policies*obs_policy + self.place_actions*obs_actions
			# EMBED STATE ACTION POLICY
			obs_actions_policies_embed = self.state_act_pol_embed_list[i](obs_actions_policies)
			obs_actions_policies_embed = obs_actions_policies_embed.repeat(1,self.num_agents,1,1).reshape(obs_actions_policies_embed.shape[0],self.num_agents,self.num_agents,self.num_agents,-1)
			weight = weight.unsqueeze(-2).repeat(1,1,self.num_agents,1).unsqueeze(-1)
			weighted_attention_values = obs_actions_policies_embed*weight
			attention_values_list.append(torch.sum(weighted_attention_values, dim=-2))

		node_features = torch.cat(attention_values_list, dim=-1)
		Value = F.leaky_relu(self.final_value_layer_1(node_features))
		Value = self.final_value_layer_2(Value)

		return Value, weights_list


class SemiHardGATV2Critic(nn.Module):
	'''
	https://arxiv.org/pdf/2105.14491.pdf
	'''
	def __init__(self, obs_input_dim, obs_output_dim, obs_act_input_dim, obs_act_output_dim, final_input_dim, final_output_dim, num_agents, num_actions, weight_threshold=None, kth_weight=None):
		super(SemiHardGATV2Critic, self).__init__()

		self.name = "SemiHardGATV2Critic"
		
		self.num_agents = num_agents
		self.num_actions = num_actions
		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		# self.device = "cpu"
		self.weight_threshold = weight_threshold
		self.kth_weight = kth_weight

		self.state_embed = nn.Sequential(nn.Linear(obs_input_dim*2, obs_output_dim), nn.LeakyReLU())
		self.state_act_pol_embed = nn.Sequential(nn.Linear(obs_act_input_dim, obs_act_output_dim), nn.LeakyReLU())
		self.attention_network = nn.Linear(obs_output_dim, 1)

		# NOISE
		self.noise_normal = torch.distributions.Normal(loc=torch.tensor([0.0]), scale=torch.tensor([1.0]))
		self.noise_uniform = torch.rand
		# ********************************************************************************************************

		# ********************************************************************************************************
		# FCN FINAL LAYER TO GET VALUES
		self.final_value_layer_1 = nn.Linear(final_input_dim, 64, bias=False)
		self.final_value_layer_2 = nn.Linear(64, final_output_dim, bias=False)
		# ********************************************************************************************************	

		self.place_policies = torch.zeros(self.num_agents,self.num_agents,obs_act_input_dim).to(self.device)
		self.place_actions = torch.ones(self.num_agents,self.num_agents,obs_act_input_dim).to(self.device)
		one_hots = torch.ones(obs_act_input_dim)
		zero_hots = torch.zeros(obs_act_input_dim)

		for j in range(self.num_agents):
			self.place_policies[j][j] = one_hots
			self.place_actions[j][j] = zero_hots


		self.reset_parameters()


	def reset_parameters(self):
		"""Reinitialize learnable parameters."""
		gain_leaky = nn.init.calculate_gain('leaky_relu')

		nn.init.xavier_uniform_(self.state_embed[0].weight, gain=gain_leaky)
		nn.init.xavier_uniform_(self.state_act_pol_embed[0].weight, gain=gain_leaky)

		nn.init.xavier_uniform_(self.attention_network.weight)

		nn.init.xavier_uniform_(self.final_value_layer_1.weight, gain=gain_leaky)
		nn.init.xavier_uniform_(self.final_value_layer_2.weight, gain=gain_leaky)


	def forward(self, states, policies, actions):

		source_state = states.unsqueeze(2).repeat(1,1,self.num_agents,1)
		destination_state = states.unsqueeze(1).repeat(1,self.num_agents,1,1)
		# WEIGHT
		state_embed_input = torch.cat([source_state, destination_state], dim=-1)
		states_embed = self.state_embed(state_embed_input)
		weight = F.softmax(self.attention_network(states_embed).squeeze(-1),dim=-1)

		if self.kth_weight is not None:
			weight_sub, indices = torch.topk(weight,k=self.kth_weight,dim=-1)
			weight_sub = weight_sub[:,:,-1].unsqueeze(-1)
			weight = F.relu(weight-weight_sub)
			weight = F.softmax(weight,dim=-1)
		elif self.weight_threshold is not None:
			weight = F.relu(weight - self.weight_threshold)
			weight = F.softmax(weight,dim=-1)

		ret_weight = weight

		obs_actions = torch.cat([states,actions],dim=-1)
		obs_policy = torch.cat([states,policies], dim=-1)
		obs_actions = obs_actions.repeat(1,self.num_agents,1).reshape(obs_actions.shape[0],self.num_agents,self.num_agents,-1)
		obs_policy = obs_policy.repeat(1,self.num_agents,1).reshape(obs_policy.shape[0],self.num_agents,self.num_agents,-1)
		obs_actions_policies = self.place_policies*obs_policy + self.place_actions*obs_actions
		# EMBED STATE ACTION POLICY
		obs_actions_policies_embed = self.state_act_pol_embed(obs_actions_policies)
		obs_actions_policies_embed = obs_actions_policies_embed.repeat(1,self.num_agents,1,1).reshape(obs_actions_policies_embed.shape[0],self.num_agents,self.num_agents,self.num_agents,-1)
		weight = weight.unsqueeze(-2).repeat(1,1,self.num_agents,1).unsqueeze(-1)
		weighted_attention_values = obs_actions_policies_embed*weight
		node_features = torch.sum(weighted_attention_values, dim=-2)

		Value = F.leaky_relu(self.final_value_layer_1(node_features))
		Value = self.final_value_layer_2(Value)

		return Value, ret_weight



class SemiHardMultiHeadGATV2Critic(nn.Module):
	'''
	https://arxiv.org/pdf/2105.14491.pdf
	'''
	def __init__(self, obs_input_dim, obs_output_dim, obs_act_input_dim, obs_act_output_dim, final_input_dim, final_output_dim, num_agents, num_actions, weight_threshold=None, kth_weight=None, num_heads=2):
		super(SemiHardMultiHeadGATV2Critic, self).__init__()

		self.name = "SemiHardMultiHeadGATV2Critic" + str(num_heads)
		
		self.num_agents = num_agents
		self.num_actions = num_actions
		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		# self.device = "cpu"
		self.kth_weight = kth_weight
		self.weight_threshold = weight_threshold
		self.num_heads = num_heads

		self.state_embed_list = []
		self.state_act_pol_embed_list = []
		self.attention_network_list = []

		multi_head_obs_output_dim = obs_output_dim//self.num_heads
		multi_head_obs_act_output_dim = obs_act_output_dim//self.num_heads

		for i in range(self.num_heads):
			self.state_embed_list.append(nn.Sequential(nn.Linear(obs_input_dim*2, multi_head_obs_output_dim), nn.LeakyReLU()).to(self.device))
			self.state_act_pol_embed_list.append(nn.Sequential(nn.Linear(obs_act_input_dim, multi_head_obs_act_output_dim), nn.LeakyReLU()).to(self.device))
			self.attention_network_list.append(nn.Linear(multi_head_obs_output_dim, 1).to(self.device))

		# NOISE
		self.noise_normal = torch.distributions.Normal(loc=torch.tensor([0.0]), scale=torch.tensor([1.0]))
		self.noise_uniform = torch.rand
		# ********************************************************************************************************

		# ********************************************************************************************************
		# FCN FINAL LAYER TO GET VALUES
		if final_input_dim != multi_head_obs_act_output_dim*self.num_heads:
			final_input_dim = multi_head_obs_act_output_dim*self.num_heads
		self.final_value_layer_1 = nn.Linear(final_input_dim, 64, bias=False)
		self.final_value_layer_2 = nn.Linear(64, final_output_dim, bias=False)
		# ********************************************************************************************************	

		self.place_policies = torch.zeros(self.num_agents,self.num_agents,obs_act_input_dim).to(self.device)
		self.place_actions = torch.ones(self.num_agents,self.num_agents,obs_act_input_dim).to(self.device)
		one_hots = torch.ones(obs_act_input_dim)
		zero_hots = torch.zeros(obs_act_input_dim)

		for j in range(self.num_agents):
			self.place_policies[j][j] = one_hots
			self.place_actions[j][j] = zero_hots


		self.reset_parameters()


	def reset_parameters(self):
		"""Reinitialize learnable parameters."""
		gain_leaky = nn.init.calculate_gain('leaky_relu')

		for i in range(self.num_heads):
			nn.init.xavier_uniform_(self.state_embed_list[i][0].weight, gain=gain_leaky)
			nn.init.xavier_uniform_(self.state_act_pol_embed_list[i][0].weight, gain=gain_leaky)
			nn.init.xavier_uniform_(self.attention_network_list[i].weight)

		nn.init.xavier_uniform_(self.final_value_layer_1.weight, gain=gain_leaky)
		nn.init.xavier_uniform_(self.final_value_layer_2.weight, gain=gain_leaky)


	def forward(self, states, policies, actions):

		source_state = states.unsqueeze(2).repeat(1,1,self.num_agents,1)
		destination_state = states.unsqueeze(1).repeat(1,self.num_agents,1,1)
		# WEIGHT
		state_embed_input = torch.cat([source_state, destination_state], dim=-1)

		weights_list = []
		attention_values_list = []
		for i in range(self.num_heads):
			states_embed = self.state_embed_list[i](state_embed_input)
			weight = F.softmax(self.attention_network_list[i](states_embed).squeeze(-1),dim=-1)

			if self.kth_weight is not None:
				weight_sub, indices = torch.topk(weight,k=self.kth_weight,dim=-1)
				weight_sub = weight_sub[:,:,-1].unsqueeze(-1)
				weight = F.relu(weight-weight_sub)
				weight = F.softmax(weight,dim=-1)
			elif self.weight_threshold is not None:
				weight = F.relu(weight - self.weight_threshold)
				weight = F.softmax(weight,dim=-1)
			weights_list.append(weight)

			obs_actions = torch.cat([states,actions],dim=-1)
			obs_policy = torch.cat([states,policies], dim=-1)
			obs_actions = obs_actions.repeat(1,self.num_agents,1).reshape(obs_actions.shape[0],self.num_agents,self.num_agents,-1)
			obs_policy = obs_policy.repeat(1,self.num_agents,1).reshape(obs_policy.shape[0],self.num_agents,self.num_agents,-1)
			obs_actions_policies = self.place_policies*obs_policy + self.place_actions*obs_actions
			# EMBED STATE ACTION POLICY
			obs_actions_policies_embed = self.state_act_pol_embed_list[i](obs_actions_policies)
			obs_actions_policies_embed = obs_actions_policies_embed.repeat(1,self.num_agents,1,1).reshape(obs_actions_policies_embed.shape[0],self.num_agents,self.num_agents,self.num_agents,-1)
			weight = weight.unsqueeze(-2).repeat(1,1,self.num_agents,1).unsqueeze(-1)
			weighted_attention_values = obs_actions_policies_embed*weight
			attention_values_list.append(torch.sum(weighted_attention_values, dim=-2))

		node_features = torch.cat(attention_values_list, dim=-1)
		Value = F.leaky_relu(self.final_value_layer_1(node_features))
		Value = self.final_value_layer_2(Value)

		return Value, weights_list

'''
Replacing Softmax attention with Normalized attention
'''
class NormalizedAttentionTransformerCritic(nn.Module):
	'''
	https://arxiv.org/pdf/2005.09561.pdf
	'''
	def __init__(self, obs_input_dim, obs_output_dim, obs_act_input_dim, obs_act_output_dim, final_input_dim, final_output_dim, num_agents, num_actions):
		super(NormalizedAttentionTransformerCritic, self).__init__()

		self.name = "NormalizedAttentionTransformerCritic"
		
		self.num_agents = num_agents
		self.num_actions = num_actions
		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		# self.device = "cpu"

		self.state_embed = nn.Sequential(nn.Linear(obs_input_dim, 128), nn.LeakyReLU())
		self.key_layer = nn.Linear(128, obs_output_dim, bias=False)
		self.query_layer = nn.Linear(128, obs_output_dim, bias=False)
		self.state_act_pol_embed = nn.Sequential(nn.Linear(obs_act_input_dim, 128), nn.LeakyReLU())
		self.attention_value_layer = nn.Linear(128, obs_act_output_dim, bias=False)
		self.layer_norm = nn.LayerNorm([self.num_agents,self.num_agents])
		# dimesion of key
		self.d_k_obs_act = obs_output_dim  

		# Normalized weights parameters
		# self.gain = torch.nn.parameter.Parameter(torch.Tensor([1]), requires_grad=True)
		# self.bias = torch.nn.parameter.Parameter(torch.Tensor([0]), requires_grad=True)
		# self.epsilon = 1e-3

		# NOISE
		self.noise_normal = torch.distributions.Normal(loc=torch.tensor([0.0]), scale=torch.tensor([1.0]))
		self.noise_uniform = torch.rand
		# ********************************************************************************************************

		# ********************************************************************************************************
		# FCN FINAL LAYER TO GET VALUES
		self.final_value_layer_1 = nn.Linear(final_input_dim, 64, bias=False)
		self.final_value_layer_2 = nn.Linear(64, final_output_dim, bias=False)
		# ********************************************************************************************************	

		self.place_policies = torch.zeros(self.num_agents,self.num_agents,obs_act_input_dim).to(self.device)
		self.place_actions = torch.ones(self.num_agents,self.num_agents,obs_act_input_dim).to(self.device)
		one_hots = torch.ones(obs_act_input_dim)
		zero_hots = torch.zeros(obs_act_input_dim)

		for j in range(self.num_agents):
			self.place_policies[j][j] = one_hots
			self.place_actions[j][j] = zero_hots


		self.reset_parameters()


	def reset_parameters(self):
		"""Reinitialize learnable parameters."""
		gain_leaky = nn.init.calculate_gain('leaky_relu')

		nn.init.xavier_uniform_(self.state_embed[0].weight, gain=gain_leaky)
		nn.init.xavier_uniform_(self.state_act_pol_embed[0].weight, gain=gain_leaky)

		nn.init.xavier_uniform_(self.key_layer.weight)
		nn.init.xavier_uniform_(self.query_layer.weight)
		nn.init.xavier_uniform_(self.attention_value_layer.weight)


		nn.init.xavier_uniform_(self.final_value_layer_1.weight, gain=gain_leaky)
		nn.init.xavier_uniform_(self.final_value_layer_2.weight, gain=gain_leaky)

	def normalize_weights(self, scores):
		std, mean = torch.std_mean(scores, unbiased=False, dim=-1)
		std += self.epsilon
		normalize = torch.div((scores - mean.unsqueeze(-1)),std.unsqueeze(-1))
		return self.gain*normalize + self.bias

	def forward(self, states, policies, actions):
		# EMBED STATES
		states_embed = self.state_embed(states)
		# KEYS
		key_obs = self.key_layer(states_embed)
		# QUERIES
		query_obs = self.query_layer(states_embed)
		# WEIGHT
		score = torch.matmul(query_obs,key_obs.transpose(1,2))/math.sqrt(self.d_k_obs_act)
		# weight = self.normalize_weights(score)
		weight = self.layer_norm(score)
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

		Value = F.leaky_relu(self.final_value_layer_1(node_features))
		Value = self.final_value_layer_2(Value)

		return Value, ret_weight


class MultiHeadNormalizedAttentionTransformerCritic(nn.Module):
	'''
	https://arxiv.org/pdf/2005.09561.pdf
	'''
	def __init__(self, obs_input_dim, obs_output_dim, obs_act_input_dim, obs_act_output_dim, final_input_dim, final_output_dim, num_agents, num_actions, num_heads=2):
		super(MultiHeadNormalizedAttentionTransformerCritic, self).__init__()

		self.name = "MultiHeadNormalizedAttentionTransformerCritic" + str(num_heads)
		
		self.num_agents = num_agents
		self.num_actions = num_actions
		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		# self.device = "cpu"
		# self.epsilon = 1e-3
		self.num_heads = num_heads

		self.state_embed_list = []
		self.key_layer_list = []
		self.query_layer_list = []
		self.state_act_pol_embed_list = []
		self.attention_value_layer_list = []
		self.layer_norm_list = []
		# self.gain_list = []
		# self.bias_list = []
		multi_head_hidden_dim = 128//self.num_heads
		multi_head_obs_output_dim = obs_output_dim//self.num_heads
		multi_head_obs_act_output_dim = obs_act_output_dim//self.num_heads
		for i in range(self.num_heads):
			self.state_embed_list.append(nn.Sequential(nn.Linear(obs_input_dim, multi_head_hidden_dim), nn.LeakyReLU()).to(self.device))
			self.key_layer_list.append(nn.Linear(multi_head_hidden_dim, multi_head_obs_output_dim, bias=False).to(self.device))
			self.query_layer_list.append(nn.Linear(multi_head_hidden_dim, multi_head_obs_output_dim, bias=False).to(self.device))
			self.state_act_pol_embed_list.append(nn.Sequential(nn.Linear(obs_act_input_dim, multi_head_hidden_dim), nn.LeakyReLU()).to(self.device))
			self.attention_value_layer_list.append(nn.Linear(multi_head_hidden_dim, multi_head_obs_act_output_dim, bias=False).to(self.device))
			# Normalized weights parameters
			# self.gain_list.append(torch.nn.parameter.Parameter(torch.Tensor([1]), requires_grad=True).to(self.device))
			# self.bias_list.append(torch.nn.parameter.Parameter(torch.Tensor([0]), requires_grad=True).to(self.device))
			self.layer_norm_list.append(nn.LayerNorm([self.num_agents,self.num_agents]).to(self.device))
			

		# dimesion of key
		self.d_k_obs_act = multi_head_obs_act_output_dim  

		# NOISE
		self.noise_normal = torch.distributions.Normal(loc=torch.tensor([0.0]), scale=torch.tensor([1.0]))
		self.noise_uniform = torch.rand
		# ********************************************************************************************************

		# ********************************************************************************************************
		# FCN FINAL LAYER TO GET VALUES
		if final_input_dim != multi_head_obs_act_output_dim*self.num_heads:
			final_input_dim = multi_head_obs_act_output_dim*self.num_heads

		self.final_value_layer_1 = nn.Linear(final_input_dim, 64, bias=False)
		self.final_value_layer_2 = nn.Linear(64, final_output_dim, bias=False)
		# ********************************************************************************************************	

		self.place_policies = torch.zeros(self.num_agents,self.num_agents,obs_act_input_dim).to(self.device)
		self.place_actions = torch.ones(self.num_agents,self.num_agents,obs_act_input_dim).to(self.device)
		one_hots = torch.ones(obs_act_input_dim)
		zero_hots = torch.zeros(obs_act_input_dim)

		for j in range(self.num_agents):
			self.place_policies[j][j] = one_hots
			self.place_actions[j][j] = zero_hots


		self.reset_parameters()


	def reset_parameters(self):
		"""Reinitialize learnable parameters."""
		gain_leaky = nn.init.calculate_gain('leaky_relu')

		for i in range(self.num_heads):
			nn.init.xavier_uniform_(self.state_embed_list[i][0].weight, gain=gain_leaky)
			nn.init.xavier_uniform_(self.state_act_pol_embed_list[i][0].weight, gain=gain_leaky)

			nn.init.xavier_uniform_(self.key_layer_list[i].weight)
			nn.init.xavier_uniform_(self.query_layer_list[i].weight)
			nn.init.xavier_uniform_(self.attention_value_layer_list[i].weight)


		nn.init.xavier_uniform_(self.final_value_layer_1.weight, gain=gain_leaky)
		nn.init.xavier_uniform_(self.final_value_layer_2.weight, gain=gain_leaky)

	def normalize_weights(self, scores):
		std, mean = torch.std_mean(scores, unbiased=False, dim=-1)
		std += self.epsilon
		normalize = torch.div((scores - mean.unsqueeze(-1)),std.unsqueeze(-1))
		return self.gain*normalize + self.bias

	def forward(self, states, policies, actions):
		weights_list = []
		attention_values_list = []

		obs_actions = torch.cat([states,actions],dim=-1)
		obs_policy = torch.cat([states,policies], dim=-1)
		obs_actions = obs_actions.repeat(1,self.num_agents,1).reshape(obs_actions.shape[0],self.num_agents,self.num_agents,-1)
		obs_policy = obs_policy.repeat(1,self.num_agents,1).reshape(obs_policy.shape[0],self.num_agents,self.num_agents,-1)
		obs_actions_policies = self.place_policies*obs_policy + self.place_actions*obs_actions

		for i in range(self.num_heads):
			# EMBED STATES
			states_embed = self.state_embed_list[i](states)
			# KEYS
			key_obs = self.key_layer_list[i](states_embed)
			# QUERIES
			query_obs = self.query_layer_list[i](states_embed)
			# WEIGHT
			score = torch.matmul(query_obs,key_obs.transpose(1,2))/math.sqrt(self.d_k_obs_act)
			# weight = self.normalize_weights(score)
			weight = self.layer_norm_list[i](score)
			weights_list.append(weight)

		
			# EMBED STATE ACTION POLICY
			obs_actions_policies_embed = self.state_act_pol_embed_list[i](obs_actions_policies)
			attention_values = self.attention_value_layer_list[i](obs_actions_policies_embed)
			attention_values = attention_values.repeat(1,self.num_agents,1,1).reshape(attention_values.shape[0],self.num_agents,self.num_agents,self.num_agents,-1)
			
			weight = weight.unsqueeze(-2).repeat(1,1,self.num_agents,1).unsqueeze(-1)
			weighted_attention_values = attention_values*weight
			attention_values_list.append(torch.sum(weighted_attention_values, dim=-2))

		node_features = torch.cat(attention_values_list, dim=-1)

		Value = F.leaky_relu(self.final_value_layer_1(node_features))
		Value = self.final_value_layer_2(Value)

		return Value, weights_list



class SemiHardNormalizedAttentionTransformerCritic(nn.Module):
	'''
	https://arxiv.org/pdf/2005.09561.pdf
	'''
	def __init__(self, obs_input_dim, obs_output_dim, obs_act_input_dim, obs_act_output_dim, final_input_dim, final_output_dim, num_agents, num_actions):
		super(SemiHardNormalizedAttentionTransformerCritic, self).__init__()

		self.name = "SemiHardNormalizedAttentionTransformerCritic"
		
		self.num_agents = num_agents
		self.num_actions = num_actions
		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		# self.device = "cpu"

		self.state_embed = nn.Sequential(nn.Linear(obs_input_dim, 128), nn.LeakyReLU())
		self.key_layer = nn.Linear(128, obs_output_dim, bias=False)
		self.query_layer = nn.Linear(128, obs_output_dim, bias=False)
		self.state_act_pol_embed = nn.Sequential(nn.Linear(obs_act_input_dim, 128), nn.LeakyReLU())
		self.attention_value_layer = nn.Linear(128, obs_act_output_dim, bias=False)
		self.layer_norm = nn.LayerNorm([self.num_agents,self.num_agents])
		# dimesion of key
		self.d_k_obs_act = obs_output_dim  

		# Normalized weights parameters
		# self.gain = torch.nn.parameter.Parameter(torch.Tensor([1]), requires_grad=True)
		# self.bias = torch.nn.parameter.Parameter(torch.Tensor([0]), requires_grad=True)
		# self.epsilon = 1e-3

		# NOISE
		self.noise_normal = torch.distributions.Normal(loc=torch.tensor([0.0]), scale=torch.tensor([1.0]))
		self.noise_uniform = torch.rand
		# ********************************************************************************************************

		# ********************************************************************************************************
		# FCN FINAL LAYER TO GET VALUES
		self.final_value_layer_1 = nn.Linear(final_input_dim, 64, bias=False)
		self.final_value_layer_2 = nn.Linear(64, final_output_dim, bias=False)
		# ********************************************************************************************************	

		self.place_policies = torch.zeros(self.num_agents,self.num_agents,obs_act_input_dim).to(self.device)
		self.place_actions = torch.ones(self.num_agents,self.num_agents,obs_act_input_dim).to(self.device)
		one_hots = torch.ones(obs_act_input_dim)
		zero_hots = torch.zeros(obs_act_input_dim)

		for j in range(self.num_agents):
			self.place_policies[j][j] = one_hots
			self.place_actions[j][j] = zero_hots


		self.reset_parameters()


	def reset_parameters(self):
		"""Reinitialize learnable parameters."""
		gain_leaky = nn.init.calculate_gain('leaky_relu')

		nn.init.xavier_uniform_(self.state_embed[0].weight, gain=gain_leaky)
		nn.init.xavier_uniform_(self.state_act_pol_embed[0].weight, gain=gain_leaky)

		nn.init.xavier_uniform_(self.key_layer.weight)
		nn.init.xavier_uniform_(self.query_layer.weight)
		nn.init.xavier_uniform_(self.attention_value_layer.weight)


		nn.init.xavier_uniform_(self.final_value_layer_1.weight, gain=gain_leaky)
		nn.init.xavier_uniform_(self.final_value_layer_2.weight, gain=gain_leaky)

	def normalize_weights(self, scores):
		std, mean = torch.std_mean(scores, unbiased=False, dim=-1)
		std += self.epsilon
		normalize = torch.div((scores - mean.unsqueeze(-1)),std.unsqueeze(-1))
		return self.gain*normalize + self.bias

	def forward(self, states, policies, actions):
		# EMBED STATES
		states_embed = self.state_embed(states)
		# KEYS
		key_obs = self.key_layer(states_embed)
		# QUERIES
		query_obs = self.query_layer(states_embed)
		# WEIGHT
		score = F.relu(torch.matmul(query_obs,key_obs.transpose(1,2))/math.sqrt(self.d_k_obs_act))
		# weight = self.normalize_weights(score)
		weight = self.layer_norm(score)
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

		Value = F.leaky_relu(self.final_value_layer_1(node_features))
		Value = self.final_value_layer_2(Value)

		return Value, ret_weight


class SemiHardMultiHeadNormalizedAttentionTransformerCritic(nn.Module):
	'''
	https://arxiv.org/pdf/2005.09561.pdf
	'''
	def __init__(self, obs_input_dim, obs_output_dim, obs_act_input_dim, obs_act_output_dim, final_input_dim, final_output_dim, num_agents, num_actions, num_heads=2):
		super(SemiHardMultiHeadNormalizedAttentionTransformerCritic, self).__init__()

		self.name = "SemiHardMultiHeadNormalizedAttentionTransformerCritic" + str(num_heads)
		
		self.num_agents = num_agents
		self.num_actions = num_actions
		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		# self.device = "cpu"
		# self.epsilon = 1e-3
		self.num_heads = num_heads

		self.state_embed_list = []
		self.key_layer_list = []
		self.query_layer_list = []
		self.state_act_pol_embed_list = []
		self.attention_value_layer_list = []
		self.layer_norm_list = []
		# self.gain_list = []
		# self.bias_list = []
		multi_head_hidden_dim = 128//self.num_heads
		multi_head_obs_output_dim = obs_output_dim//self.num_heads
		multi_head_obs_act_output_dim = obs_act_output_dim//self.num_heads
		for i in range(self.num_heads):
			self.state_embed_list.append(nn.Sequential(nn.Linear(obs_input_dim, multi_head_hidden_dim), nn.LeakyReLU()).to(self.device))
			self.key_layer_list.append(nn.Linear(multi_head_hidden_dim, multi_head_obs_output_dim, bias=False).to(self.device))
			self.query_layer_list.append(nn.Linear(multi_head_hidden_dim, multi_head_obs_output_dim, bias=False).to(self.device))
			self.state_act_pol_embed_list.append(nn.Sequential(nn.Linear(obs_act_input_dim, multi_head_hidden_dim), nn.LeakyReLU()).to(self.device))
			self.attention_value_layer_list.append(nn.Linear(multi_head_hidden_dim, multi_head_obs_act_output_dim, bias=False).to(self.device))
			# Normalized weights parameters
			# self.gain_list.append(torch.nn.parameter.Parameter(torch.Tensor([1]), requires_grad=True).to(self.device))
			# self.bias_list.append(torch.nn.parameter.Parameter(torch.Tensor([0]), requires_grad=True).to(self.device))
			self.layer_norm_list.append(nn.LayerNorm([self.num_agents,self.num_agents]).to(self.device))
			

		# dimesion of key
		self.d_k_obs_act = multi_head_obs_act_output_dim  

		# NOISE
		self.noise_normal = torch.distributions.Normal(loc=torch.tensor([0.0]), scale=torch.tensor([1.0]))
		self.noise_uniform = torch.rand
		# ********************************************************************************************************

		# ********************************************************************************************************
		# FCN FINAL LAYER TO GET VALUES
		if final_input_dim != multi_head_obs_act_output_dim*self.num_heads:
			final_input_dim = multi_head_obs_act_output_dim*self.num_heads

		self.final_value_layer_1 = nn.Linear(final_input_dim, 64, bias=False)
		self.final_value_layer_2 = nn.Linear(64, final_output_dim, bias=False)
		# ********************************************************************************************************	

		self.place_policies = torch.zeros(self.num_agents,self.num_agents,obs_act_input_dim).to(self.device)
		self.place_actions = torch.ones(self.num_agents,self.num_agents,obs_act_input_dim).to(self.device)
		one_hots = torch.ones(obs_act_input_dim)
		zero_hots = torch.zeros(obs_act_input_dim)

		for j in range(self.num_agents):
			self.place_policies[j][j] = one_hots
			self.place_actions[j][j] = zero_hots


		self.reset_parameters()


	def reset_parameters(self):
		"""Reinitialize learnable parameters."""
		gain_leaky = nn.init.calculate_gain('leaky_relu')

		for i in range(self.num_heads):
			nn.init.xavier_uniform_(self.state_embed_list[i][0].weight, gain=gain_leaky)
			nn.init.xavier_uniform_(self.state_act_pol_embed_list[i][0].weight, gain=gain_leaky)

			nn.init.xavier_uniform_(self.key_layer_list[i].weight)
			nn.init.xavier_uniform_(self.query_layer_list[i].weight)
			nn.init.xavier_uniform_(self.attention_value_layer_list[i].weight)


		nn.init.xavier_uniform_(self.final_value_layer_1.weight, gain=gain_leaky)
		nn.init.xavier_uniform_(self.final_value_layer_2.weight, gain=gain_leaky)

	def normalize_weights(self, scores):
		std, mean = torch.std_mean(scores, unbiased=False, dim=-1)
		std += self.epsilon
		normalize = torch.div((scores - mean.unsqueeze(-1)),std.unsqueeze(-1))
		return self.gain*normalize + self.bias

	def forward(self, states, policies, actions):
		weights_list = []
		attention_values_list = []

		obs_actions = torch.cat([states,actions],dim=-1)
		obs_policy = torch.cat([states,policies], dim=-1)
		obs_actions = obs_actions.repeat(1,self.num_agents,1).reshape(obs_actions.shape[0],self.num_agents,self.num_agents,-1)
		obs_policy = obs_policy.repeat(1,self.num_agents,1).reshape(obs_policy.shape[0],self.num_agents,self.num_agents,-1)
		obs_actions_policies = self.place_policies*obs_policy + self.place_actions*obs_actions

		for i in range(self.num_heads):
			# EMBED STATES
			states_embed = self.state_embed_list[i](states)
			# KEYS
			key_obs = self.key_layer_list[i](states_embed)
			# QUERIES
			query_obs = self.query_layer_list[i](states_embed)
			# WEIGHT
			score = F.relu(torch.matmul(query_obs,key_obs.transpose(1,2))/math.sqrt(self.d_k_obs_act))
			# weight = self.normalize_weights(score)
			weight = self.layer_norm_list[i](score)
			weights_list.append(weight)

		
			# EMBED STATE ACTION POLICY
			obs_actions_policies_embed = self.state_act_pol_embed_list[i](obs_actions_policies)
			attention_values = self.attention_value_layer_list[i](obs_actions_policies_embed)
			attention_values = attention_values.repeat(1,self.num_agents,1,1).reshape(attention_values.shape[0],self.num_agents,self.num_agents,self.num_agents,-1)
			
			weight = weight.unsqueeze(-2).repeat(1,1,self.num_agents,1).unsqueeze(-1)
			weighted_attention_values = attention_values*weight
			attention_values_list.append(torch.sum(weighted_attention_values, dim=-2))

		node_features = torch.cat(attention_values_list, dim=-1)

		Value = F.leaky_relu(self.final_value_layer_1(node_features))
		Value = self.final_value_layer_2(Value)

		return Value, weights_list