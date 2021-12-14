from typing import Any, List, Tuple, Union
import torch
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
		self.states_critic = []
		self.states_actor = []
		self.logprobs = []
		self.rewards = []
		self.dones = []
	

	def clear(self):
		del self.actions[:]
		del self.states_critic[:]
		del self.states_actor[:]
		del self.probs[:]
		del self.one_hot_actions[:]
		del self.logprobs[:]
		del self.rewards[:]
		del self.dones[:]


class MLPPolicy(nn.Module):
	def __init__(self,current_agent_state_dim, other_agent_state_dim, num_agents, action_dim, device):
		super(MLPPolicy,self).__init__()

		self.name = "MLPPolicy"
		self.num_agents = num_agents		
		self.device = device

		self.Policy = nn.Sequential(
			nn.Linear(current_agent_state_dim+(other_agent_state_dim*(num_agents-1)),128),
			nn.LeakyReLU(),
			# nn.Linear(128,128),
			# nn.LeakyReLU(),
			nn.Linear(128,64),
			nn.LeakyReLU(),
			nn.Linear(64,action_dim),
			nn.Softmax(dim=-1)
			)

	def reset_parameters(self):
		gain_leaky = nn.init.calculate_gain('leaky_relu')

		nn.init.orthogonal_(self.Policy[0].weight, gain=gain_leaky)	
		nn.init.orthogonal_(self.Policy[2].weight, gain=gain_leaky)		
		nn.init.orthogonal_(self.Policy[4].weight, gain=gain_leaky)
		# nn.init.orthogonal_(self.Policy[6].weight, gain=gain_leaky)	

	def forward(self, states):

		# T x num_agents x state_dim
		T = states.shape[0]
		
		# # [s0;s1;s2;s3]  -> [s0 s1 s2 s3; s1 s2 s3 s0; s2 s3 s1 s0 ....]

		# states_aug = [torch.roll(states,i,1) for i in range(self.num_agents)]

		# states_aug = torch.cat(states_aug,dim=2)

		Policy = self.Policy(states)

		return Policy, 1/self.num_agents*torch.ones((T,self.num_agents,self.num_agents),device=self.device)


class TransformerPolicy(nn.Module):
	def __init__(self, obs_input_dim, final_output_dim, num_agents, num_actions, num_heads, device):
		super(TransformerPolicy, self).__init__()
		
		self.name = "TransformerPolicy"
		self.num_heads = num_heads

		self.num_agents = num_agents
		self.num_actions = num_actions
		self.device = device

		obs_output_dim = 128//self.num_heads

		self.state_embed_list = []
		self.key_list = []
		self.query_list = []
		self.attention_value_list = []
		for i in range(self.num_heads):
			self.state_embed_list.append(nn.Sequential(nn.Linear(obs_input_dim, 128), nn.LeakyReLU()).to(self.device))
			self.key_list.append(nn.Linear(128, obs_output_dim, bias=True).to(self.device))
			self.query_list.append(nn.Linear(128, obs_output_dim, bias=True).to(self.device))
			self.attention_value_list.append(nn.Sequential(nn.Linear(128, obs_output_dim, bias=True), nn.LeakyReLU()).to(self.device))

		self.d_k = obs_output_dim
		# ********************************************************************************************************

		# ********************************************************************************************************
		# FCN FINAL LAYER TO GET VALUES
		final_input_dim = obs_output_dim*self.num_heads
		self.final_policy_layers = nn.Sequential(
			nn.Linear(final_input_dim, 64, bias=True), 
			nn.LeakyReLU(),
			nn.Linear(64, final_output_dim, bias=True)
			)
		# ********************************************************************************************************


		self.reset_parameters()


	def reset_parameters(self):
		"""Reinitialize learnable parameters."""
		gain_leaky = nn.init.calculate_gain('leaky_relu')

		for i in range(self.num_heads):
			nn.init.orthogonal_(self.state_embed_list[i][0].weight, gain=gain_leaky)

			nn.init.orthogonal_(self.key_list[i].weight)
			nn.init.orthogonal_(self.query_list[i].weight)
			nn.init.orthogonal_(self.attention_value_list[i][0].weight)


		nn.init.orthogonal_(self.final_policy_layers[0].weight, gain=gain_leaky)
		nn.init.orthogonal_(self.final_policy_layers[2].weight, gain=gain_leaky)



	def forward(self, states):
		weights = []
		node_features = []
		for i in range(self.num_heads):
			# EMBED STATES
			states_embed = self.state_embed_list[i](states)
			# KEYS
			key_obs = self.key_list[i](states_embed)
			# QUERIES
			query_obs = self.query_list[i](states_embed)
			# WEIGHT
			weight = F.softmax(torch.matmul(query_obs,key_obs.transpose(1,2))/math.sqrt(self.d_k),dim=-1)
			weights.append(weight)

			attention_values = self.attention_value_list[i](states_embed)
			node_feature = torch.matmul(weight, attention_values)

			node_features.append(node_feature)

		node_features = torch.cat(node_features, dim=-1).to(self.device)
		Policy = F.softmax(self.final_policy_layers(node_features), dim=-1)

		return Policy, weights




# using Q network of MAAC
class Q_network(nn.Module):
	'''
	https://proceedings.neurips.cc/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf
	'''
	def __init__(self, obs_input_dim, final_output_dim, num_agents, num_actions, device):
		super(Q_network, self).__init__()
		
		self.name = "Q_network"

		self.num_agents = num_agents
		self.num_actions = num_actions
		self.device = device

		obs_output_dim = 128
		obs_act_input_dim = obs_input_dim+self.num_actions
		obs_act_output_dim = 128

		self.state_embed_query = nn.Sequential(nn.Linear(obs_input_dim, 128, bias=True), nn.LeakyReLU())
		self.state_embed_key = nn.Sequential(nn.Linear(obs_input_dim, 128, bias=True), nn.LeakyReLU())
		self.key = nn.Linear(128, obs_output_dim, bias=True)
		self.query = nn.Linear(128, obs_output_dim, bias=True)
		
		self.state_act_embed = nn.Sequential(nn.Linear(obs_act_input_dim, 128, bias=True), nn.LeakyReLU())
		self.attention_value = nn.Sequential(nn.Linear(128, obs_act_output_dim, bias=True), nn.LeakyReLU())
		self.curr_agent_state_embed = nn.Sequential(nn.Linear(obs_input_dim, 64, bias=True), nn.LeakyReLU()).to(self.device)

		

		# dimesion of key
		self.d_k = obs_output_dim

		# ********************************************************************************************************

		# ********************************************************************************************************
		final_input_dim = obs_act_output_dim + 64
		# FCN FINAL LAYER TO GET VALUES
		self.final_value_layers = nn.Sequential(
			nn.Linear(final_input_dim, 64, bias=True), 
			nn.LeakyReLU(),
			nn.Linear(64, final_output_dim, bias=True)
			)
		# ********************************************************************************************************


		self.reset_parameters()


	def reset_parameters(self):
		"""Reinitialize learnable parameters."""
		gain_leaky = nn.init.calculate_gain('leaky_relu')

		nn.init.orthogonal_(self.state_embed_query[0].weight, gain=gain_leaky)
		nn.init.orthogonal_(self.state_embed_key[0].weight, gain=gain_leaky)
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


	def forward(self, states, policies, actions):

		states_query = states.unsqueeze(-2)
		states_key = states.unsqueeze(1).repeat(1,self.num_agents,1,1)
		states_key = self.remove_self_loops(states_key)
		actions_ = self.remove_self_loops(actions.unsqueeze(1).repeat(1,self.num_agents,1,1))

		obs_actions = torch.cat([states_key,actions_],dim=-1)

		# EMBED STATES QUERY
		states_query_embed = self.state_embed_query(states_query)
		# EMBED STATES QUERY
		states_key_embed = self.state_embed_key(states_key)
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