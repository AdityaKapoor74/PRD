from typing import Any, List, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import datetime
import math


'''
Scalar Dot Product Attention: Transformer
'''

class TransformerRewardPredictor(nn.Module):
	'''
	https://proceedings.neurips.cc/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf
	'''
	def __init__(self, obs_input_dim, obs_output_dim, final_input_dim, final_output_dim, num_agents, num_actions):
		super(TransformerRewardPredictor, self).__init__()
		
		self.name = "TransformerRewardPredictor"

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
		self.final_value_layer_1 = nn.Linear(final_input_dim, 64, bias=False)
		self.final_value_layer_2 = nn.Linear(64, final_output_dim, bias=False)
		# ********************************************************************************************************	


		self.reset_parameters()


	def reset_parameters(self):
		"""Reinitialize learnable parameters."""
		gain_leaky = nn.init.calculate_gain('leaky_relu')

		nn.init.xavier_uniform_(self.state_embed[0].weight, gain=gain_leaky)

		nn.init.xavier_uniform_(self.key_layer.weight)
		nn.init.xavier_uniform_(self.query_layer.weight)
		nn.init.xavier_uniform_(self.attention_value_layer.weight)


		nn.init.xavier_uniform_(self.final_value_layer_1.weight, gain=gain_leaky)
		nn.init.xavier_uniform_(self.final_value_layer_2.weight, gain=gain_leaky)



	def forward(self, states):
		# EMBED STATES
		states_embed = self.state_embed(states)
		# KEYS
		key_obs = self.key_layer(states_embed)
		# QUERIES
		query_obs = self.query_layer(states_embed)
		# WEIGHT
		weight = F.softmax(torch.matmul(query_obs,key_obs.transpose(1,2))/math.sqrt(self.d_k_obs_act),dim=-1)
		# ATTENTION VALUES
		attention_values = self.attention_value_layer(states_embed)
		node_features = torch.matmul(weight, attention_values)

		indiv_rewards = F.leaky_relu(self.final_value_layer_1(node_features))
		indiv_rewards = self.final_value_layer_2(indiv_rewards)

		shared_reward = torch.sum(indiv_rewards, dim=-2)

		return shared_reward, indiv_rewards, weight



class JointRewardPredictor(nn.Module):
	'''
	https://proceedings.neurips.cc/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf
	'''
	def __init__(self, obs_act_input_dim, obs_act_output_dim, final_input_dim, final_output_dim):
		super(JointRewardPredictor, self).__init__()
		
		self.name = "JointRewardPredictor"
		
		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		# self.device = "cpu"

		self.state_act_embed = nn.Sequential(nn.Linear(obs_act_input_dim, 128), nn.LeakyReLU())
		self.attention_weight = nn.Sequential(nn.Linear(128, 64), nn.LeakyReLU(), nn.Linear(64, 1))
		self.node_feature_embed = nn.Sequential(nn.Linear(128, 64), nn.LeakyReLU(), nn.Linear(64, 64))

		# NOISE
		self.noise_normal = torch.distributions.Normal(loc=torch.tensor([0.0]), scale=torch.tensor([1.0]))
		self.noise_uniform = torch.rand
		# ********************************************************************************************************

		# ********************************************************************************************************
		# FCN FINAL LAYER TO GET VALUES
		self.reward_predictor = nn.Sequential(nn.Linear(final_input_dim, 64), nn.LeakyReLU(), nn.Linear(64, final_output_dim))
		# ********************************************************************************************************	


		self.reset_parameters()


	def reset_parameters(self):
		"""Reinitialize learnable parameters."""
		gain_leaky = nn.init.calculate_gain('leaky_relu')

		nn.init.xavier_uniform_(self.state_act_embed[0].weight, gain=gain_leaky)

		nn.init.xavier_uniform_(self.attention_weight[0].weight, gain=gain_leaky)
		nn.init.xavier_uniform_(self.attention_weight[2].weight, gain=gain_leaky)

		nn.init.xavier_uniform_(self.node_feature_embed[0].weight, gain=gain_leaky)
		nn.init.xavier_uniform_(self.node_feature_embed[2].weight, gain=gain_leaky)


		nn.init.xavier_uniform_(self.reward_predictor[0].weight, gain=gain_leaky)
		nn.init.xavier_uniform_(self.reward_predictor[2].weight, gain=gain_leaky)



	def forward(self, states, actions):
		state_actions = torch.cat([states, actions], dim=-1)
		# EMBED STATES ACTIONS
		states_act_embed = self.state_act_embed(state_actions)
		# FEATURE VECTOR
		feature_vector = self.node_feature_embed(states_act_embed)
		# ATTENTION VALUES
		alpha = self.attention_weight(states_act_embed)
		# WEIGHT
		weights = F.softmax(alpha,dim=-2)
		# NODE FEATURE
		node_features = torch.sum(weights*feature_vector, dim=-2)
		# REWARD PREDICTION
		Reward = self.reward_predictor(node_features)

		return Reward, weights