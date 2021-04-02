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





class SoftAttentionInput(nn.Module):
	def __init__(self, in_dim, out_dim,num_agents):
		super(SoftAttentionInput, self).__init__()
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

		# nn.init.xavier_uniform_(self.query_fc_layer.weight)
		# nn.init.xavier_uniform_(self.value_fc_layer.weight)
		# nn.init.xavier_uniform_(self.key_fc_layer.weight)


	def message_func(self, edges):
		return {'value': edges.src['value'], 'score': ((edges.src['key'] * edges.dst['query']).sum(-1, keepdim=True))}

	def reduce_func(self, nodes):
		# reduce UDF for equation (3) & (4)
		# equation (3)
		# alpha = torch.softmax(torch.exp((nodes.mailbox['score'] / math.sqrt(self.d_k)).clamp(-5, 5)), dim=-2)
		alpha = torch.sigmoid(nodes.mailbox['score'] / math.sqrt(self.d_k))
		# equation (4)
		obs_proc = torch.sum(alpha * nodes.mailbox['value'], dim=1)
		
		# with open('../../weights/Experiment10/'+f"{datetime.datetime.now():%d-%m-%Y}"+'preprocessed_obs_no_relu.txt','a+') as f:
		# 	torch.set_printoptions(profile="full")
		# 	print("*"*100,file=f)
		# 	print("PROCESSED OBSERVATIONS",file=f)
		# 	print(obs_proc,file=f)	
		# 	print("*"*100,file=f)
		# 	print("WEIGHTS",file=f)
		# 	print(alpha,file=f)
		# 	print("*"*100,file=f)
		# 	torch.set_printoptions(profile="default")
		
		return {'obs_proc': obs_proc, "weights":alpha}

	def forward(self, g, observations):
		self.g = g
		key = torch.tanh(self.key_fc_layer_1(observations))
		key = self.key_fc_layer_2(key)
		# key = self.key_fc_layer(observations)
		query = torch.tanh(self.query_fc_layer_1(observations))
		query = self.query_fc_layer_2(query)
		# query = self.query_fc_layer(observations)
		value = torch.tanh(self.value_fc_layer_1(observations))
		value = self.value_fc_layer_2(value)
		# value = self.value_fc_layer(observations)
		self.g.ndata['value'] = value
		self.g.ndata['key'] = key
		self.g.ndata['query'] = query
		self.g.update_all(self.message_func, self.reduce_func)
		return self.g.ndata.pop('obs_proc'), self.g.ndata.pop('weights')



class SoftAttentionWeight(nn.Module):
	def __init__(self, in_dim, out_dim, num_agents,num_actions):
		super(SoftAttentionWeight, self).__init__()
		self.num_agents = num_agents
		self.num_actions = num_actions
		# self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		self.device = "cpu"
		self.key_fc_layer_1 = nn.Linear(in_dim, 32, bias=True)
		self.key_fc_layer_2 = nn.Linear(32, out_dim, bias=True)
		# self.key_fc_layer = nn.Linear(in_dim, out_dim, bias=True)

		self.query_fc_layer_1 = nn.Linear(in_dim, 32, bias=True)
		self.query_fc_layer_2 = nn.Linear(32, out_dim, bias=True)
		# self.query_fc_layer = nn.Linear(in_dim, out_dim, bias=True)

		# dimesion of key
		self.d_k = out_dim

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
		nn.init.xavier_uniform_(self.key_fc_layer_1.weight)
		nn.init.xavier_uniform_(self.key_fc_layer_2.weight)
		nn.init.xavier_uniform_(self.query_fc_layer_1.weight)
		nn.init.xavier_uniform_(self.query_fc_layer_2.weight)
		# nn.init.xavier_uniform_(self.key_fc_layer.weight)
		# nn.init.xavier_uniform_(self.query_fc_layer.weight)


	def message_func(self, edges):
		return {'score': ((edges.src['key'] * edges.dst['query']).sum(-1, keepdim=True)), 'pi': edges.src['pi'], 'act': edges.src['act']}

	def reduce_func(self, nodes):
		# reduce UDF for equation (3)
		# equation (3)
		# w = torch.softmax(torch.exp((nodes.mailbox['score'] / math.sqrt(self.d_k)).clamp(-5, 5)), dim=-2)
		w = torch.sigmoid(nodes.mailbox['score'] / math.sqrt(self.d_k))
		z = w*nodes.mailbox['act'] + (1-w)*nodes.mailbox['pi']
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
		key = torch.tanh(self.key_fc_layer_1(h))
		key = self.key_fc_layer_2(key)
		# key = self.key_fc_layer(h)
		query = torch.tanh(self.query_fc_layer_1(h))
		query = self.query_fc_layer_2(query)
		# query = self.query_fc_layer(h)
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
		# self.input_processor = GATLayerInput(preprocess_input_dim, preprocess_output_dim, num_agents)
		# self.weight_layer = GATLayer(weight_input_dim, weight_output_dim, num_agents, num_actions)
		# SCALAR DOT ATTENTION
		self.input_processor = SoftAttentionInput(preprocess_input_dim, preprocess_output_dim, num_agents)
		self.weight_layer = SoftAttentionWeight(weight_input_dim, weight_output_dim, num_agents, num_actions)
		self.value_layer = ValueNetwork(input_dim, output_dim)

	def forward(self, g, policies, actions):
		features, weights_preproc = self.input_processor(g, g.ndata['obs'])
		g.ndata['obs_proc'] = features
		obs_final, weights = self.weight_layer(g,g.ndata['mypose_goalpose'],policies,actions)
		x = self.value_layer(obs_final)
		return x, weights, weights_preproc





class CriticNetwork(nn.Module):
	def __init__(self, weight_input_dim, weight_output_dim, obs_z_input_dim, obs_z_output_dim, final_input_dim, final_output_dim, num_agents, num_actions):
		super(CriticNetwork, self).__init__()
		
		self.num_agents = num_agents
		self.num_actions = num_actions
		# self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		self.device = "cpu"
		self.z = None
		self.weight_z = None

		# *******************************************************************************************************
		# WEIGHT NETWORK
		self.key_weight_layer_1 = nn.Linear(weight_input_dim, 64, bias=True)
		self.key_weight_layer_2 = nn.Linear(64, weight_output_dim, bias=True)
		# self.key_fc_layer = nn.Linear(weight_input_dim, weight_output_dim, bias=True)

		self.query_weight_layer_1 = nn.Linear(weight_input_dim, 64, bias=True)
		self.query_weight_layer_2 = nn.Linear(64, weight_output_dim, bias=True)
		# self.query_fc_layer = nn.Linear(weight_input_dim, weight_output_dim, bias=True)

		# dimesion of key
		self.d_k_weight = weight_output_dim
		# ********************************************************************************************************

		# *******************************************************************************************************
		# PROCESSING OF OBSERVATIONS
		self.key_obsz_layer_1 = nn.Linear(obs_z_input_dim, 64, bias=True)
		self.key_obsz_layer_2 = nn.Linear(64, obs_z_output_dim, bias=True)
		# self.key_fc_layer = nn.Linear(obs_z_input_dim, obs_z_output_dim, bias=True)

		self.query_obsz_layer_1 = nn.Linear(obs_z_input_dim, 64, bias=True)
		self.query_obsz_layer_2 = nn.Linear(64, obs_z_output_dim, bias=True)
		# self.query_fc_layer = nn.Linear(obs_z_input_dim, obs_z_output_dim, bias=True)

		self.attention_value_obsz_layer_1 = nn.Linear(obs_z_input_dim, 64, bias=True)
		self.attention_value_obsz_layer_2 = nn.Linear(64, obs_z_output_dim, bias=True)
		# self.attention_value_obsz_layer = nn.Linear(obs_z_input_dim, obs_z_output_dim, bias=True)


		# dimesion of key
		self.d_k_obsz = obs_z_output_dim
		# ********************************************************************************************************

		# ********************************************************************************************************
		# FCN FINAL LAYER TO GET VALUES
		self.final_value_layer_1 = nn.Linear(final_input_dim, 64, bias=True)
		self.final_value_layer_2 = nn.Linear(64, final_output_dim, bias=True)
		# ********************************************************************************************************

		# ********************************************************************************************************
		# Extracting source nodes observation,z values
		self.source_obsz = torch.zeros(self.num_agents,self.num_agents,obs_z_input_dim).to(self.device)
		one_hots = torch.ones(obs_z_input_dim)
		for j in range(self.num_agents):
			self.source_obsz[j][j] = one_hots
		# ********************************************************************************************************	

		# ********************************************************************************************************
		# Placing Policies instead of zs
		self.place_policies = torch.zeros(self.num_agents,self.num_agents,self.num_agents,obs_z_input_dim).to(self.device)
		self.place_zs = torch.ones(self.num_agents,self.num_agents,self.num_agents,obs_z_input_dim).to(self.device)
		one_hots = torch.ones(obs_z_input_dim)
		zero_hots = torch.zeros(obs_z_input_dim)

		for i in range(self.num_agents):
			for j in range(self.num_agents):
				self.place_policies[i][j][j] = one_hots
				self.place_zs[i][j][j] = zero_hots

		# self.place_policies = self.place_policies.reshape(self.num_agents,-1,obs_z_input_dim)
		# self.place_zs = self.place_zs.reshape(self.num_agents,-1,obs_z_input_dim)

		# ********************************************************************************************************* 

		self.reset_parameters()

	def reset_parameters(self):
		"""Reinitialize learnable parameters."""
		gain = nn.init.calculate_gain('leaky_relu')

		nn.init.xavier_uniform_(self.key_weight_layer_1.weight, gain=gain)
		nn.init.xavier_uniform_(self.key_weight_layer_2.weight, gain=gain)
		nn.init.xavier_uniform_(self.query_weight_layer_1.weight, gain=gain)
		nn.init.xavier_uniform_(self.query_weight_layer_2.weight, gain=gain)

		nn.init.xavier_uniform_(self.key_obsz_layer_1.weight, gain=gain)
		nn.init.xavier_uniform_(self.key_obsz_layer_2.weight, gain=gain)
		nn.init.xavier_uniform_(self.query_obsz_layer_1.weight, gain=gain)
		nn.init.xavier_uniform_(self.query_obsz_layer_2.weight, gain=gain)
		nn.init.xavier_uniform_(self.attention_value_obsz_layer_1.weight, gain=gain)
		nn.init.xavier_uniform_(self.attention_value_obsz_layer_2.weight, gain=gain)

		nn.init.xavier_uniform_(self.final_value_layer_1.weight, gain=gain)
		nn.init.xavier_uniform_(self.final_value_layer_2.weight, gain=gain)


	def weight_message_func(self, edges):
		return {'score': ((edges.src['weight_key'] * edges.dst['weight_query']).sum(-1, keepdim=True)), 'pi': edges.src['pi'], 'act': edges.src['act']}

	def weight_reduce_func(self, nodes):

		w = torch.sigmoid(nodes.mailbox['score'] / math.sqrt(self.d_k_weight))
		z = w*nodes.mailbox['act'] + (1-w)*nodes.mailbox['pi']

		return {'z':z, 'w': w}


	def forward(self, g, policies, actions):
		# equation (1)
		self.g = g
		key_z = F.leaky_relu(self.key_weight_layer_1(g.ndata['obs']))
		key_z = self.key_weight_layer_2(key_z)

		query_z = F.leaky_relu(self.query_weight_layer_1(g.ndata['obs']))
		query_z = self.query_weight_layer_2(query_z)

		self.g.ndata['weight_key'] = key_z
		self.g.ndata['weight_query'] = query_z

		self.g.ndata['pi'] = policies.reshape(-1,self.num_actions)
		self.g.ndata['act'] = actions.reshape(-1,self.num_actions)

		self.g.update_all(self.weight_message_func, self.weight_reduce_func)

		z = self.g.ndata.pop('z') 
		weight_z = self.g.ndata.pop('w')

		obs = g.ndata['obs'].reshape(g.ndata['obs'].shape[0]//self.num_agents,self.num_agents,-1)
		obs = obs.repeat(1,self.num_agents,1).reshape(obs.shape[0],self.num_agents,self.num_agents,-1)
		z = z.reshape(z.shape[0]//self.num_agents,self.num_agents,self.num_agents,-1)

		obs_z = torch.cat([obs,z],dim=-1) #KEYS

		obs_z_shape = obs_z.shape
		source_obsz = self.source_obsz
		source_obsz = source_obsz.repeat(obs_z.shape[0],1,1).reshape(obs_z_shape)

		# storing obs_z for every node --> obs_1, z_11; obs_2, z_21 ...
		source_obs_z = torch.sum(source_obsz * obs_z, dim=-2) #to match dimensions
		source_obs_z_ = source_obs_z # saving to calculate source observationzs
		source_obs_z = source_obs_z.repeat(1,1,self.num_agents).reshape(obs_z_shape) #QUERIES

		key_obsz = F.leaky_relu(self.key_obsz_layer_1(source_obs_z))
		key_obsz = self.key_obsz_layer_2(key_obsz)

		query_obsz = F.leaky_relu(self.query_obsz_layer_1(obs_z))
		query_obsz = self.query_obsz_layer_2(query_obsz)

		# inflating the dimensions to include policies so that we can calculate V(i,j) = Estimated return for agent i not conditioned on agent j's actions
		obs_z = obs_z.repeat(1,1,self.num_agents,1).reshape(obs_z.shape[0],self.num_agents,self.num_agents,self.num_agents,-1)
		# doing these operations to get timesteps x num_agents x num_agents x num_agents x dimension
		place_policies = torch.cat([self.g.ndata['obs'],self.g.ndata['pi']],dim=-1).reshape(obs_z.shape[0],self.num_agents,-1)
		place_policies = place_policies.repeat(1,self.num_agents,1).reshape(place_policies.shape[0],self.num_agents,self.num_agents,-1)
		place_policies = place_policies.repeat(1,self.num_agents,1,1).reshape(place_policies.shape[0],self.num_agents,self.num_agents,self.num_agents,-1)
		obs_zs = obs_z*self.place_zs
		obs_pis = place_policies*self.place_policies
		obs_z_pi = obs_zs + obs_pis

		attention_value_src_obsz = F.leaky_relu(self.attention_value_obsz_layer_1(source_obs_z_))
		attention_value_src_obsz = self.attention_value_obsz_layer_2(attention_value_src_obsz)

		attention_value_other_obsz = F.leaky_relu(self.attention_value_obsz_layer_1(obs_z_pi))
		attention_value_other_obsz = self.attention_value_obsz_layer_2(attention_value_other_obsz)

		score_obsz = torch.sum(key_obsz * query_obsz, dim=-1)
		weights_obsz = torch.softmax(torch.exp((score_obsz / math.sqrt(self.d_k_obsz)).clamp(-5, 5)), dim=-1).unsqueeze(-1)
		weights_obsz = weights_obsz.repeat(1,self.num_agents,1,1).reshape(weights_obsz.shape[0],self.num_agents,self.num_agents,self.num_agents,-1)

		# # CASE 1, Value = aggregation of all attention_value_other_obsz
		# node_features = torch.mean(attention_value_other_obsz * weights_obsz, dim=-2)

		# # CASE 2, Value = attention_value_src_obsz, aggregation of all attention_value_other_obsz
		attention_value_src_obsz = attention_value_src_obsz.repeat(1,self.num_agents,1).reshape(attention_value_src_obsz.shape[0],self.num_agents,self.num_agents,-1)
		node_features = torch.cat([attention_value_src_obsz,torch.mean(attention_value_other_obsz * weights_obsz, dim=-2)], dim=-1)

		Value = F.leaky_relu(self.final_value_layer_1(node_features))
		Value = self.final_value_layer_2(Value)


		return Value, weight_z, weights_obsz



class PolicyNetwork(nn.Module):
	def __init__(
		self,
		policy_sizes
		):
		super(PolicyNetwork,self).__init__()

		self.policy = create_model(policy_sizes)

	def forward(self,states):
		return F.softmax(self.policy(states),-1)