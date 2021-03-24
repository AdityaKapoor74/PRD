from typing import Any, List, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl 
import numpy as np
import dgl
import dgl.function as fn
from dgl import DGLGraph
import datetime
import math

# *******************************************
# Q(s,a) 
# *******************************************

def create_model(
	layer_sizes: Tuple,
	weight_init: str = "xavier_uniform",
	activation_func: str = "leaky_relu"
	):

	layers = []
	limit = len(layer_sizes)

	# add more activations
	if activation_func == "tanh":
		activation = nn.Tanh()
	elif activation_func == "relu":
		activation = nn.ReLU()
	elif activation_func == "leaky_relu":
		activation = nn.LeakyReLU()

	# add more weight init
	if weight_init == "xavier_uniform":
		weight_init = torch.nn.init.xavier_uniform_
	elif weight_init == "xavier_normal":
		weight_init = torch.nn.init.xavier_normal_

	for layer in range(limit - 1):
		act = activation if layer < limit - 2 else nn.Identity()
		layers += [nn.Linear(layer_sizes[layer], layer_sizes[layer + 1])]
		weight_init(layers[-1].weight)
		layers += [act]

	return nn.Sequential(*layers)



class SoftAttentionInput_9_1(nn.Module):
	def __init__(self, in_dim, out_dim,num_agents):
		super(SoftAttentionInput_9_1, self).__init__()
		# equation (1)
		self.key_fc_layer_1 = nn.Linear(in_dim, 32, bias=True)
		self.key_fc_layer_2 = nn.Linear(32, out_dim, bias=True)
		# self.key_fc_layer = nn.Linear(in_dim, out_dim, bias=True)

		self.query_fc_layer_1 = nn.Linear(in_dim, 32, bias=True)
		self.query_fc_layer_2 = nn.Linear(32, out_dim, bias=True)
		# self.query_fc_layer = nn.Linear(in_dim, out_dim, bias=True)

		self.value_fc_layer_1 = nn.Linear(in_dim, 32, bias=True)
		self.value_fc_layer_2 = nn.Linear(32, out_dim, bias=True)
		# self.value_fc_layer = nn.Linear(in_dim, out_dim, bias=True)

		# output dim of key
		self.d_k = out_dim

		self.num_agents = num_agents
		self.agent_pairing = torch.zeros(self.num_agents,self.num_agents)

		self.reset_parameters()

	def reset_parameters(self):
		"""Reinitialize learnable parameters."""
		# gain = nn.init.calculate_gain('leaky_relu')
		nn.init.xavier_uniform_(self.key_fc_layer_1.weight)
		nn.init.xavier_uniform_(self.key_fc_layer_2.weight)
		nn.init.xavier_uniform_(self.query_fc_layer_1.weight)
		nn.init.xavier_uniform_(self.query_fc_layer_2.weight)
		nn.init.xavier_uniform_(self.value_fc_layer_1.weight)
		nn.init.xavier_uniform_(self.value_fc_layer_2.weight)
		# nn.init.xavier_uniform_(self.key_fc_layer.weight)
		# nn.init.xavier_uniform_(self.query_fc_layer.weight)
		# nn.init.xavier_uniform_(self.value_fc_layer.weight)


	def message_func(self, edges):
		return {'value': edges.src['value'], 'score': ((edges.src['key'] * edges.dst['query']).sum(-1, keepdim=True))}

	def reduce_func(self, nodes):
		# reduce UDF for equation (3) & (4)
		# equation (3)
		# alpha = torch.sigmoid(nodes.mailbox['score'] / math.sqrt(self.d_k))
		# alpha = torch.softmax(torch.exp((nodes.mailbox['score'] / math.sqrt(self.d_k)).clamp(-5, 5)), dim=-2)
		alpha = torch.softmax(nodes.mailbox['score'] / math.sqrt(self.d_k), dim=-2)
		# equation (4)
		obs_proc = torch.sum(alpha * nodes.mailbox['value'], dim=1)
		
		return {'obs_proc': obs_proc, "weights":alpha}

	def forward(self, g, observations):
		self.g = g
		key = torch.tanh(self.key_fc_layer_1(observations))
		key = self.key_fc_layer_2(key)
		# key = self.key_fc_layer(observations)
		query = torch.tanh(self.query_fc_layer_1(observations))
		query = self.query_fc_layer_2(query)
		# query = self.query_fc_layer(observations)
		values = torch.tanh(self.value_fc_layer_1(observations))
		values = self.value_fc_layer_2(values)
		# values = self.value_fc_layer(observations)

		self.g.ndata['value'] = values
		self.g.ndata['key'] = key
		self.g.ndata['query'] = query

		self.g.update_all(self.message_func, self.reduce_func)
		return self.g.ndata.pop('obs_proc'), self.g.ndata.pop('weights')



class SoftAttentionWeight_9_1(nn.Module):
	def __init__(self, in_dim, out_dim, num_agents,num_actions):
		super(SoftAttentionWeight_9_1, self).__init__()
		self.num_agents = num_agents
		self.num_actions = num_actions
		# self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		self.device = "cpu"

		# store weights
		self.w = None

		# add noise
		self.normal_dist = torch.distributions.Normal(loc=torch.tensor([0.]), scale=torch.tensor([0.1]))

		self.place_policies = torch.zeros(self.num_agents,self.num_agents,self.num_agents,num_actions).to(self.device)
		self.place_zs = torch.ones(self.num_agents,self.num_agents,self.num_agents,num_actions).to(self.device)
		one_hots = torch.ones(num_actions)
		zero_hots = torch.zeros(num_actions)

		for i in range(self.num_agents):
			for j in range(self.num_agents):
				self.place_policies[i][j][j] = one_hots
				self.place_zs[i][j][j] = zero_hots

		self.place_policies = self.place_policies.reshape(self.num_agents,-1,num_actions)
		self.place_zs = self.place_zs.reshape(self.num_agents,-1,num_actions)


	def message_func(self, edges):
		return {'pi': edges.src['pi'], 'act': edges.src['act']}

	def reduce_func(self, nodes):
		# reduce UDF for equation (3)
		# equation (3)
		noise = self.normal_dist.sample((nodes.mailbox['act'].view(-1).size())).reshape(nodes.mailbox['act'].size())
		z = self.w*nodes.mailbox['act'] + (1-self.w)*nodes.mailbox['pi'] + noise 
		z = z.repeat(1,self.num_agents,1)
		pi = nodes.mailbox['pi'].repeat(1,self.num_agents,1).reshape(-1,self.place_policies.shape[0],self.place_policies.shape[1],self.place_policies.shape[2])*self.place_policies
		zs = z.reshape(-1,self.place_zs.shape[0],self.place_zs.shape[1],self.place_zs.shape[2])*self.place_zs
		z = (pi+zs)
		z = z.reshape(z.shape[0],z.shape[1],self.num_agents,self.num_agents,self.num_actions)
		z = torch.mean(z,dim=-2)
		obs_proc = self.g.ndata['obs_proc'].reshape(-1,self.num_agents,self.g.ndata['obs_proc'].shape[1]).repeat(1,self.num_agents,1)
		obs_proc = obs_proc.reshape(obs_proc.shape[0],self.num_agents,self.num_agents,-1)
		
		obs_final = torch.cat([obs_proc.reshape(-1,obs_proc.shape[-1]),z.reshape(-1,self.num_actions)],dim=-1).reshape(obs_proc.shape[0]*obs_proc.shape[1],obs_proc.shape[2],-1)

		return {'obs_final':obs_final}

	def forward(self, g, policies, actions, weights):
		# equation (1)
		self.w = weights
		self.g = g

		self.g.ndata['pi'] = policies.reshape(-1,self.num_actions)
		self.g.ndata['act'] = actions.reshape(-1,self.num_actions)
		# equation (3) & (4)
		self.g.update_all(self.message_func, self.reduce_func)
		return self.g.ndata.pop('obs_final')


class SoftAttentionWeight_9_1_(nn.Module):
	def __init__(self, in_dim, out_dim, num_agents,num_actions):
		super(SoftAttentionWeight_9_1_, self).__init__()
		self.num_agents = num_agents
		self.num_actions = num_actions
		# self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		self.device = "cpu"
		# self.key_fc_layer_1 = nn.Linear(in_dim, 32, bias=True)
		# self.key_fc_layer_2 = nn.Linear(32, out_dim, bias=True)
		self.key_fc_layer = nn.Linear(in_dim, out_dim, bias=True)

		# self.query_fc_layer_1 = nn.Linear(in_dim, 32, bias=True)
		# self.query_fc_layer_2 = nn.Linear(32, out_dim, bias=True)
		self.query_fc_layer = nn.Linear(in_dim, out_dim, bias=True)

		# dimesion of query
		self.d_k = out_dim

		# add noise
		self.normal_dist = torch.distributions.Normal(loc=torch.tensor([0.]), scale=torch.tensor([0.1]))

		self.place_policies = torch.zeros(self.num_agents,self.num_agents,self.num_agents,num_actions).to(self.device)
		self.place_zs = torch.ones(self.num_agents,self.num_agents,self.num_agents,num_actions).to(self.device)
		one_hots = torch.ones(num_actions)
		zero_hots = torch.zeros(num_actions)

		for i in range(self.num_agents):
			for j in range(self.num_agents):
				self.place_policies[i][j][j] = one_hots
				self.place_zs[i][j][j] = zero_hots

		self.place_policies = self.place_policies.reshape(self.num_agents,-1,num_actions)
		self.place_zs = self.place_zs.reshape(self.num_agents,-1,num_actions)


		self.reset_parameters()

	def reset_parameters(self):
		"""Reinitialize learnable parameters."""
		# gain = nn.init.calculate_gain('leaky_relu')
		# nn.init.xavier_uniform_(self.key_fc_layer_1.weight)
		# nn.init.xavier_uniform_(self.key_fc_layer_2.weight)
		# nn.init.xavier_uniform_(self.query_fc_layer_1.weight)
		# nn.init.xavier_uniform_(self.query_fc_layer_2.weight)
		nn.init.xavier_uniform_(self.key_fc_layer.weight)
		nn.init.xavier_uniform_(self.query_fc_layer.weight)


	def message_func(self, edges):
		return {'score': ((edges.src['key'] * edges.dst['query']).sum(-1, keepdim=True)), 'pi': edges.src['pi'], 'act': edges.src['act']}

	def reduce_func(self, nodes):
		# reduce UDF for equation (3)
		# equation (3)
		# w = torch.sigmoid(nodes.mailbox['score'] / math.sqrt(self.d_k))
		w = torch.softmax(nodes.mailbox['score'] / math.sqrt(self.d_k), dim=-2)
		noise = self.normal_dist.sample((nodes.mailbox['act'].view(-1).size())).reshape(nodes.mailbox['act'].size())
		z = w*nodes.mailbox['act'] + (1-w)*nodes.mailbox['pi'] + noise
		z = z.repeat(1,self.num_agents,1)
		pi = nodes.mailbox['pi'].repeat(1,self.num_agents,1).reshape(-1,self.place_policies.shape[0],self.place_policies.shape[1],self.place_policies.shape[2])*self.place_policies
		zs = z.reshape(-1,self.place_zs.shape[0],self.place_zs.shape[1],self.place_zs.shape[2])*self.place_zs
		z = (pi+zs)
		z = z.reshape(z.shape[0],z.shape[1],self.num_agents,self.num_agents,self.num_actions)
		z = torch.mean(z,dim=-2)
		obs_proc = self.g.ndata['obs_proc'].reshape(-1,self.num_agents,self.g.ndata['obs_proc'].shape[1]).repeat(1,self.num_agents,1)
		obs_proc = obs_proc.reshape(obs_proc.shape[0],self.num_agents,self.num_agents,-1)
		
		obs_final = torch.cat([obs_proc.reshape(-1,obs_proc.shape[-1]),z.reshape(-1,self.num_actions)],dim=-1).reshape(obs_proc.shape[0]*obs_proc.shape[1],obs_proc.shape[2],-1)

		return {'obs_final':obs_final, 'w': w}

	def forward(self, g, h, policies, actions):
		# equation (1)
		self.g = g
		# key = torch.tanh(self.key_fc_layer_1(h))
		# key = self.key_fc_layer_2(key)
		key = self.key_fc_layer(h)

		# query = torch.tanh(self.query_fc_layer_1(h))
		# query = self.query_fc_layer_2(query)
		query = self.query_fc_layer(h)

		self.g.ndata['key'] = key
		self.g.ndata['query'] = query

		self.g.ndata['pi'] = policies.reshape(-1,self.num_actions)
		self.g.ndata['act'] = actions.reshape(-1,self.num_actions)
		# equation (3) & (4)
		self.g.update_all(self.message_func, self.reduce_func)
		return self.g.ndata.pop('obs_final'), self.g.ndata.pop('w')



class ValueNetwork(nn.Module):
	def __init__(
		self,
		input_dim,
		output_dim
		):
		super(ValueNetwork,self).__init__()

		self.value = nn.Linear(input_dim, output_dim)
		torch.nn.init.xavier_uniform_(self.value.weight)

	def forward(self,states):
		return self.value(states)





class CriticNetwork(nn.Module):
	def __init__(self, preprocess_input_dim, preprocess_output_dim, weight_input_dim, weight_output_dim, input_dim, output_dim, num_agents, num_actions):
		super(CriticNetwork, self).__init__()
		# SCALAR DOT ATTENTION
		self.input_processor = SoftAttentionInput_9_1(preprocess_input_dim, preprocess_output_dim, num_agents)
		self.weight_layer = SoftAttentionWeight_9_1(weight_input_dim, weight_output_dim, num_agents, num_actions)
		self.value_layer = ValueNetwork(input_dim, output_dim)

	def forward(self, g, policies, actions, weights):
		features, weights_preproc = self.input_processor(g, g.ndata['obs'])
		g.ndata['obs_proc'] = features
		obs_final = self.weight_layer(g,policies,actions,weights)
		x = self.value_layer(obs_final)
		return x, weights_preproc



class CriticNetwork_(nn.Module):
	def __init__(self, preprocess_input_dim, preprocess_output_dim, weight_input_dim, weight_output_dim, input_dim, output_dim, num_agents, num_actions):
		super(CriticNetwork_, self).__init__()
		# SCALAR DOT ATTENTION
		self.input_processor = SoftAttentionInput_9_1(preprocess_input_dim, preprocess_output_dim, num_agents)
		self.weight_layer = SoftAttentionWeight_9_1_(weight_input_dim, weight_output_dim, num_agents, num_actions)
		self.value_layer = ValueNetwork(input_dim, output_dim)

	def forward(self, g, policies, actions):
		features, weights_preproc = self.input_processor(g, g.ndata['obs'])
		g.ndata['obs_proc'] = features
		obs_final, weights = self.weight_layer(g,g.ndata['mypose_goalpose'],policies,actions)
		x = self.value_layer(obs_final)
		return x, weights, weights_preproc

class PolicyNetwork(nn.Module):
	def __init__(
		self,
		policy_sizes
		):
		super(PolicyNetwork,self).__init__()

		self.policy = create_model(policy_sizes)

	def forward(self,states):
		return F.softmax(self.policy(states),-1)