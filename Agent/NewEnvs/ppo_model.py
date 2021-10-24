from typing import Any, List, Tuple, Union
import numpy as np
import datetime
import math
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T


class RolloutBuffer:
	def __init__(self):
		self.actions = []
		self.one_hot_actions = []
		self.probs = []
		self.states = []
		self.logprobs = []
		self.rewards = []
		self.dones = []
	

	def clear(self):
		del self.actions[:]
		del self.states[:]
		del self.probs[:]
		del self.one_hot_actions[:]
		del self.logprobs[:]
		del self.rewards[:]
		del self.dones[:]



class CNNPolicyBN(nn.Module):
	def __init__(self, num_channels, num_actions, num_agents, scaling):

		super(CNNPolicyBN, self).__init__()

		self.name = "CNNPolicyBN"

		self.num_agents = num_agents
		self.num_actions = num_actions
		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		# self.device = "cpu"
		self.scaling = scaling


		self.CNN = nn.Sequential(
			nn.Conv2d(num_channels, 32, kernel_size=2, stride=1),
			nn.BatchNorm2d(32),
			nn.ReLU(),
			nn.Conv2d(32, 64, kernel_size=2, stride=1),
			nn.BatchNorm2d(64),
			nn.ReLU(),
			nn.Conv2d(64, 64, kernel_size=2, stride=1),
			nn.BatchNorm2d(64),
			nn.ReLU(),
			)

		self.Policy = nn.Sequential(
			nn.Linear(4 * 4 * 64, 512),
			nn.LeakyReLU(),
			nn.Linear(512, 128),
			nn.LeakyReLU(),
			nn.Linear(128, 64),
			nn.LeakyReLU(),
			nn.Linear(64, num_actions)
			)

	def reset_parameters(self):
		torch.nn.init.xavier_uniform_(self.CNN[0].weight)
		torch.nn.init.xavier_uniform_(self.CNN[3].weight)
		torch.nn.init.xavier_uniform_(self.CNN[6].weight)

		torch.nn.init.xavier_uniform_(self.Policy[0].weight)
		torch.nn.init.xavier_uniform_(self.Policy[2].weight)
		torch.nn.init.xavier_uniform_(self.Policy[4].weight)
		torch.nn.init.xavier_uniform_(self.Policy[6].weight)
		
	def forward(self, local_images):

		local_images = local_images.float() / self.scaling
		local_image_embeddings = self.CNN(local_images)
		if local_image_embeddings.shape[0] == 1:
			local_image_embeddings = local_image_embeddings.reshape(local_image_embeddings.shape[0], -1)
		else:
			local_image_embeddings = local_image_embeddings.reshape(local_image_embeddings.shape[0]//self.num_agents, self.num_agents,-1)
		# T x num_agents x state_dim
		T = local_image_embeddings.shape[0]
		x = self.Policy(local_image_embeddings)
		Policy = F.softmax(x, dim=-1)

		return Policy, 1/self.num_agents*torch.ones((T,self.num_agents,self.num_agents),device=self.device)



class CNNPolicy(nn.Module):
	def __init__(self, num_channels, num_actions, num_agents, scaling):

		super(CNNPolicy, self).__init__()

		self.name = "CNNPolicy"

		self.num_agents = num_agents
		self.num_actions = num_actions
		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		# self.device = "cpu"
		self.scaling = scaling


		self.CNN = nn.Sequential(
			nn.Conv2d(num_channels, 32, kernel_size=2, stride=1),
			nn.ReLU(),
			nn.Conv2d(32, 64, kernel_size=2, stride=1),
			nn.ReLU(),
			nn.Conv2d(64, 64, kernel_size=2, stride=1),
			nn.ReLU(),
			)

		self.Policy = nn.Sequential(
			nn.Linear(4 * 4 * 64, 512),
			nn.LeakyReLU(),
			# nn.Linear(512, 128),
			# nn.LeakyReLU(),
			# nn.Linear(128, 64),
			# nn.LeakyReLU(),
			nn.Linear(512, num_actions)
			)

	def reset_parameters(self):
		torch.nn.init.xavier_uniform_(self.CNN[0].weight)
		torch.nn.init.xavier_uniform_(self.CNN[2].weight)
		torch.nn.init.xavier_uniform_(self.CNN[4].weight)

		torch.nn.init.xavier_uniform_(self.Policy[0].weight)
		torch.nn.init.xavier_uniform_(self.Policy[2].weight)
		# torch.nn.init.xavier_uniform_(self.Policy[4].weight)
		# torch.nn.init.xavier_uniform_(self.Policy[6].weight)
		
	def forward(self, local_images):

		local_images = local_images.float() / self.scaling
		local_image_embeddings = self.CNN(local_images)
		if local_image_embeddings.shape[0] == 1:
			local_image_embeddings = local_image_embeddings.reshape(local_image_embeddings.shape[0], -1)
		else:
			local_image_embeddings = local_image_embeddings.reshape(local_image_embeddings.shape[0]//self.num_agents, self.num_agents,-1)
		# T x num_agents x state_dim
		T = local_image_embeddings.shape[0]
		x = self.Policy(local_image_embeddings)
		Policy = F.softmax(x, dim=-1)

		return Policy, 1/self.num_agents*torch.ones((T,self.num_agents,self.num_agents),device=self.device)



class CNNTransformerCriticBN(nn.Module):
	def __init__(self, num_channels, num_agents, num_actions, scaling):
		super(CNNTransformerCriticBN, self).__init__()
		
		self.name = "CNNTransformerCriticBN"

		self.num_agents = num_agents
		self.num_actions = num_actions
		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		# self.device = "cpu"
		self.scaling = scaling


		self.CNN = nn.Sequential(
			nn.Conv2d(num_channels, 32, kernel_size=2, stride=1),
			nn.BatchNorm2d(32),
			nn.ReLU(),
			nn.Conv2d(32, 64, kernel_size=2, stride=1),
			nn.BatchNorm2d(64),
			nn.ReLU(),
			nn.Conv2d(64, 64, kernel_size=2, stride=1),
			nn.BatchNorm2d(64),
			nn.ReLU(),
			)

		self.CNN_post = nn.Sequential(
			nn.Linear(4 * 4 * 64, 512),
			nn.LeakyReLU(),
			nn.Linear(512, 128),
			nn.LeakyReLU(),
			nn.Linear(128, 128),
			nn.LeakyReLU()
			)


		self.state_embed = nn.Sequential(nn.Linear(128, 128), nn.LeakyReLU())
		self.key_layer = nn.Linear(128, 128, bias=False)
		self.query_layer = nn.Linear(128, 128, bias=False)
		self.state_act_pol_embed = nn.Sequential(nn.Linear(128+self.num_actions, 128), nn.LeakyReLU())
		self.attention_value_layer = nn.Linear(128, 128, bias=False)
		# dimesion of key
		self.d_k_obs_act = 128  

		# NOISE
		self.noise_normal = torch.distributions.Normal(loc=torch.tensor([0.0]), scale=torch.tensor([1.0]))
		self.noise_uniform = torch.rand
		# ********************************************************************************************************

		# ********************************************************************************************************
		# FCN FINAL LAYER TO GET VALUES
		self.final_value_layers = nn.Sequential(
			nn.Linear(128, 64, bias=False),
			nn.LeakyReLU(),
			nn.Linear(64, 1, bias=False),
			)
		# ********************************************************************************************************	

		self.place_policies = torch.zeros(self.num_agents,self.num_agents,128+self.num_actions).to(self.device)
		self.place_actions = torch.ones(self.num_agents,self.num_agents,128+self.num_actions).to(self.device)
		one_hots = torch.ones(128+self.num_actions)
		zero_hots = torch.zeros(128+self.num_actions)

		for j in range(self.num_agents):
			self.place_policies[j][j] = one_hots
			self.place_actions[j][j] = zero_hots


		self.reset_parameters()


	def reset_parameters(self):
		"""Reinitialize learnable parameters."""
		gain_leaky = nn.init.calculate_gain('leaky_relu')

		torch.nn.init.xavier_uniform_(self.CNN[0].weight)
		torch.nn.init.xavier_uniform_(self.CNN[3].weight)
		torch.nn.init.xavier_uniform_(self.CNN[6].weight)

		torch.nn.init.xavier_uniform_(self.CNN_post[0].weight)
		torch.nn.init.xavier_uniform_(self.CNN_post[2].weight)
		torch.nn.init.xavier_uniform_(self.CNN_post[4].weight)

		nn.init.xavier_uniform_(self.state_embed[0].weight, gain=gain_leaky)
		nn.init.xavier_uniform_(self.state_act_pol_embed[0].weight, gain=gain_leaky)

		nn.init.xavier_uniform_(self.key_layer.weight)
		nn.init.xavier_uniform_(self.query_layer.weight)
		nn.init.xavier_uniform_(self.attention_value_layer.weight)


		nn.init.xavier_uniform_(self.final_value_layers[0].weight, gain=gain_leaky)
		nn.init.xavier_uniform_(self.final_value_layers[2].weight, gain=gain_leaky)



	def forward(self, local_images, policies, actions):
		local_images = local_images.float() / self.scaling
		local_image_embeddings = self.CNN(local_images)
		local_image_embeddings = local_image_embeddings.reshape(local_image_embeddings.shape[0]//self.num_agents, self.num_agents, -1)
		states = self.CNN_post(local_image_embeddings)

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





class CNNTransformerCritic(nn.Module):
	def __init__(self, num_channels, num_agents, num_actions, scaling):
		super(CNNTransformerCritic, self).__init__()
		
		self.name = "CNNTransformerCritic"

		self.num_agents = num_agents
		self.num_actions = num_actions
		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		# self.device = "cpu"
		self.scaling = scaling

		self.CNN = nn.Sequential(
			nn.Conv2d(num_channels, 32, kernel_size=2, stride=1),
			nn.ReLU(),
			nn.Conv2d(32, 64, kernel_size=2, stride=1),
			nn.ReLU(),
			nn.Conv2d(64, 64, kernel_size=2, stride=1),
			nn.ReLU(),
			)

		self.CNN_post = nn.Sequential(
			nn.Linear(4 * 4 * 64, 128),
			nn.LeakyReLU(),
			# nn.Linear(512, 128),
			# nn.LeakyReLU(),
			# nn.Linear(128, 128),
			# nn.LeakyReLU()
			)


		self.state_embed = nn.Sequential(nn.Linear(128, 128), nn.LeakyReLU())
		self.key_layer = nn.Linear(128, 128, bias=False)
		self.query_layer = nn.Linear(128, 128, bias=False)
		self.state_act_pol_embed = nn.Sequential(nn.Linear(128+self.num_actions, 128), nn.LeakyReLU())
		self.attention_value_layer = nn.Linear(128, 128, bias=False)
		# dimesion of key
		self.d_k_obs_act = 128  

		# NOISE
		self.noise_normal = torch.distributions.Normal(loc=torch.tensor([0.0]), scale=torch.tensor([1.0]))
		self.noise_uniform = torch.rand
		# ********************************************************************************************************

		# ********************************************************************************************************
		# FCN FINAL LAYER TO GET VALUES
		self.final_value_layers = nn.Sequential(
			nn.Linear(128, 64, bias=False),
			nn.LeakyReLU(),
			nn.Linear(64, 1, bias=False),
			)
		# ********************************************************************************************************	

		self.place_policies = torch.zeros(self.num_agents,self.num_agents,128+self.num_actions).to(self.device)
		self.place_actions = torch.ones(self.num_agents,self.num_agents,128+self.num_actions).to(self.device)
		one_hots = torch.ones(128+self.num_actions)
		zero_hots = torch.zeros(128+self.num_actions)

		for j in range(self.num_agents):
			self.place_policies[j][j] = one_hots
			self.place_actions[j][j] = zero_hots


		self.reset_parameters()


	def reset_parameters(self):
		"""Reinitialize learnable parameters."""
		gain_leaky = nn.init.calculate_gain('leaky_relu')

		torch.nn.init.xavier_uniform_(self.CNN[0].weight)
		torch.nn.init.xavier_uniform_(self.CNN[2].weight)
		torch.nn.init.xavier_uniform_(self.CNN[4].weight)

		torch.nn.init.xavier_uniform_(self.CNN_post[0].weight)
		# torch.nn.init.xavier_uniform_(self.CNN_post[2].weight)
		# torch.nn.init.xavier_uniform_(self.CNN_post[4].weight)

		nn.init.xavier_uniform_(self.state_embed[0].weight, gain=gain_leaky)
		nn.init.xavier_uniform_(self.state_act_pol_embed[0].weight, gain=gain_leaky)

		nn.init.xavier_uniform_(self.key_layer.weight)
		nn.init.xavier_uniform_(self.query_layer.weight)
		nn.init.xavier_uniform_(self.attention_value_layer.weight)


		nn.init.xavier_uniform_(self.final_value_layers[0].weight, gain=gain_leaky)
		nn.init.xavier_uniform_(self.final_value_layers[2].weight, gain=gain_leaky)



	def forward(self, local_images, policies, actions):
		local_images = local_images.float() / self.scaling
		local_image_embeddings = self.CNN(local_images)
		local_image_embeddings = local_image_embeddings.reshape(local_image_embeddings.shape[0],-1)
		states = self.CNN_post(local_image_embeddings)

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
