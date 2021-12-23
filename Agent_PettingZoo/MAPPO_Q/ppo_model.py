from typing import Any, List, Tuple, Union
import torch
import torchvision.models as models
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import datetime
import math

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


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
        
    def forward(self, x):
        return x


class CNNPolicy(nn.Module):
	def __init__(self, num_channels, num_actions, num_agents, scaling, device):

		super(CNNPolicy, self).__init__()

		self.name = "CNNPolicy"

		self.num_agents = num_agents
		self.num_actions = num_actions
		self.device = device
		self.scaling = scaling

		# self.CNN = models.resnet18(pretrained=True)
		# self.CNN.fc = Identity()

		# self.Policy = nn.Sequential(
		# 	nn.Linear(512,512),
		# 	nn.LeakyReLU(),
		# 	nn.Linear(512,256),
		# 	nn.LeakyReLU(),
		# 	nn.Linear(256,64),
		# 	nn.LeakyReLU(),
		# 	nn.Linear(64,action_dim),
		# 	nn.Softmax(dim=-1)
		# 	)

		self.CNN = nn.Sequential(
			nn.Conv2d(num_channels, 32, kernel_size=2, stride=1),
			nn.LeakyReLU(),
			nn.Conv2d(32, 64, kernel_size=2, stride=1),
			nn.LeakyReLU(),
			nn.Conv2d(64, 64, kernel_size=2, stride=1),
			nn.LeakyReLU(),
			)
			
		self.Policy = nn.Sequential(
			nn.Linear(4 * 4 * 64, 256),
			nn.LeakyReLU(),
			nn.Linear(256, 128),
			nn.LeakyReLU(),
			nn.Linear(128, 64),
			nn.LeakyReLU(),
			nn.Linear(64, num_actions),
			nn.Softmax(dim=-1)
			)

	def reset_parameters(self):
		gain_leaky = nn.init.calculate_gain('leaky_relu')
		gain_relu = nn.init.calculate_gain('relu')

		nn.init.orthogonal_(self.CNN[0].weight, gain=gain_leaky)
		nn.init.orthogonal_(self.CNN[2].weight, gain=gain_leaky)
		nn.init.orthogonal_(self.CNN[4].weight, gain=gain_leaky)

		nn.init.orthogonal_(self.Policy[0].weight, gain=gain_leaky)
		nn.init.orthogonal_(self.Policy[2].weight, gain=gain_leaky)
		nn.init.orthogonal_(self.Policy[4].weight, gain=gain_leaky)
		nn.init.orthogonal_(self.Policy[6].weight, gain=gain_leaky)

	def forward(self, local_images):
		local_images = local_images.float() / self.scaling
		cnn_input = local_images.reshape(-1, local_images.shape[2], local_images.shape[3], local_images.shape[4])
		local_image_embeddings = self.CNN(cnn_input)
		if local_image_embeddings.shape[0] == 1:
			local_image_embeddings = local_image_embeddings.reshape(local_image_embeddings.shape[0], -1)
		else:
			local_image_embeddings = local_image_embeddings.reshape(local_image_embeddings.shape[0]//self.num_agents, self.num_agents,-1)
		# T x num_agents x state_dim
		T = local_image_embeddings.shape[0]
		Policy = self.Policy(local_image_embeddings)

		return Policy, 1/self.num_agents*torch.ones((T,self.num_agents,self.num_agents),device=self.device)


# using Q network of MAAC
class CNN_Q_network(nn.Module):
	'''
	https://proceedings.neurips.cc/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf
	'''
	def __init__(self, num_channels, num_agents, num_actions, scaling, device):
		super(CNN_Q_network, self).__init__()
		
		self.num_agents = num_agents
		self.num_actions = num_actions
		self.device = device
		self.scaling = scaling

		self.CNN = nn.Sequential(
			nn.Conv2d(num_channels, 32, kernel_size=2, stride=1),
			nn.LeakyReLU(),
			nn.Conv2d(32, 64, kernel_size=2, stride=1),
			nn.LeakyReLU(),
			nn.Conv2d(64, 64, kernel_size=2, stride=1),
			nn.LeakyReLU(),
			)

		self.FC = nn.Sequential(
			nn.Linear(4 * 4 * 64, 512),
			nn.LeakyReLU(),
			nn.Linear(512, 128),
			nn.LeakyReLU(),
			)

		obs_input_dim = 128
		obs_output_dim = 128
		obs_act_input_dim = obs_input_dim+self.num_actions
		obs_act_output_dim = 128

		self.state_embed = nn.Sequential(nn.Linear(obs_input_dim, 128, bias=True), nn.LeakyReLU())
		self.key = nn.Linear(128, obs_output_dim, bias=True)
		self.query = nn.Linear(128, obs_output_dim, bias=True)
		
		self.state_act_embed = nn.Sequential(nn.Linear(obs_act_input_dim, 128, bias=True), nn.LeakyReLU())
		self.attention_value = nn.Sequential(nn.Linear(128, obs_act_output_dim, bias=True), nn.LeakyReLU())

		self.curr_agent_state_embed = nn.Sequential(nn.Linear(obs_input_dim, 64, bias=True), nn.LeakyReLU())

		# dimesion of key
		self.d_k = obs_output_dim

		# ********************************************************************************************************

		# ********************************************************************************************************
		final_input_dim = obs_act_output_dim + 64
		# FCN FINAL LAYER TO GET VALUES
		self.final_value_layers = nn.Sequential(
			nn.Linear(final_input_dim, 64, bias=True), 
			nn.LeakyReLU(),
			nn.Linear(64, self.num_actions, bias=True)
			)
		# ********************************************************************************************************


		self.reset_parameters()


	def reset_parameters(self):
		"""Reinitialize learnable parameters."""
		gain_leaky = nn.init.calculate_gain('leaky_relu')
		gain_relu = nn.init.calculate_gain('relu')

		nn.init.orthogonal_(self.CNN[0].weight, gain=gain_leaky)
		nn.init.orthogonal_(self.CNN[2].weight, gain=gain_leaky)
		nn.init.orthogonal_(self.CNN[4].weight, gain=gain_leaky)

		nn.init.orthogonal_(self.FC[0].weight, gain=gain_leaky)
		nn.init.orthogonal_(self.FC[2].weight, gain=gain_leaky)

		nn.init.orthogonal_(self.state_embed[0].weight, gain=gain_leaky)
		nn.init.orthogonal_(self.state_act_embed[0].weight, gain=gain_leaky)

		nn.init.orthogonal_(self.key.weight)
		nn.init.orthogonal_(self.query.weight)
		nn.init.orthogonal_(self.attention_value[0].weight)

		nn.init.orthogonal_(self.curr_agent_state_embed[0].weight, gain=gain_leaky)


		nn.init.orthogonal_(self.final_value_layers[0].weight, gain=gain_leaky)
		nn.init.orthogonal_(self.final_value_layers[2].weight, gain=gain_leaky)


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


	def forward(self, local_images, policies, actions):

		local_images = local_images.float() / self.scaling
		cnn_input = local_images.reshape(-1, local_images.shape[2], local_images.shape[3], local_images.shape[4])
		local_image_embeddings = self.CNN(cnn_input)
		local_image_embeddings = local_image_embeddings.reshape(local_image_embeddings.shape[0]//self.num_agents, self.num_agents, -1)
		states = self.FC(local_image_embeddings)

		states_query = states.unsqueeze(-2)
		states_key = states.unsqueeze(1).repeat(1,self.num_agents,1,1)
		states_key = self.remove_self_loops(states_key)
		actions_ = self.remove_self_loops(actions.unsqueeze(1).repeat(1,self.num_agents,1,1))

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
		weight = F.softmax(torch.matmul(query_obs,key_obs.transpose(2,3))/math.sqrt(self.d_k),dim=-1)
		weights = self.weight_assignment(weight.squeeze(-2))

		# EMBED STATE ACTION POLICY
		obs_actions_embed = self.state_act_embed(obs_actions)
		attention_values = self.attention_value(obs_actions_embed)
		node_features = torch.matmul(weight, attention_values)


		curr_agent_state_embed = self.curr_agent_state_embed(states)
		curr_agent_node_features = torch.cat([curr_agent_state_embed, node_features.squeeze(-2)], dim=-1)
		
		Q_value = self.final_value_layers(curr_agent_node_features)

		Value = torch.matmul(Q_value,policies.transpose(1,2))

		Q_value = torch.sum(actions*Q_value, dim=-1).unsqueeze(-1)

		return Value, Q_value, weights