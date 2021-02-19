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





class GATLayerInput(nn.Module):
	def __init__(self, in_dim, out_dim,num_agents):
		super(GATLayerInput, self).__init__()
		# equation (1)
		self.fc = nn.Linear(in_dim, out_dim, bias=True)
		# equation (2)
		self.attn_fc = nn.Linear(2 * out_dim, 1, bias=True)

		self.num_agents = num_agents
		self.agent_pairing = torch.zeros(self.num_agents,self.num_agents)

		self.reset_parameters()

	def reset_parameters(self):
		"""Reinitialize learnable parameters."""
		# gain = nn.init.calculate_gain('leaky_relu')
		nn.init.xavier_uniform_(self.fc.weight)
		nn.init.xavier_uniform_(self.attn_fc.weight)

	def edge_attention(self, edges):
		# edge UDF for equation (2)
		features_ = torch.cat([edges.src['features'], edges.dst['features']], dim=1)
		a = self.attn_fc(features_)
		return {'e': a}

	def message_func(self, edges):
		# message UDF for equation (3) & (4)
		return {'features': edges.src['features'], 'e': edges.data['e']}

	def reduce_func(self, nodes):
		# reduce UDF for equation (3) & (4)
		# equation (3)
		alpha = torch.sigmoid(nodes.mailbox['e'])
		# equation (4)
		obs_proc = torch.sum(alpha * nodes.mailbox['features'], dim=1)
		
		with open('../../weights/Experiment7_7/'+f"{datetime.datetime.now():%d-%m-%Y}"+'preprocessed_obs.txt','a+') as f:
			torch.set_printoptions(profile="full")
			print("*"*100,file=f)
			print("PROCESSED OBSERVATIONS",file=f)
			print(obs_proc,file=f)	
			print("*"*100,file=f)
			print("WEIGHTS",file=f)
			print(alpha,file=f)
			print("*"*100,file=f)
			torch.set_printoptions(profile="default")
		
		return {'obs_proc': obs_proc}

	def forward(self, g, observations):
		self.g = g
		# equation (1)
		features = self.fc(observations)
		self.g.ndata['features'] = features
		# equation (2)
		self.g.apply_edges(self.edge_attention)
		# equation (3) & (4)
		self.g.update_all(self.message_func, self.reduce_func)
		return self.g.ndata.pop('obs_proc')



class GATLayer(nn.Module):
	def __init__(self, in_dim, out_dim, num_agents,num_actions):
		super(GATLayer, self).__init__()
		self.num_agents = num_agents
		self.num_actions = num_actions
		# self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		self.device = "cpu"
		# equation (1)
		# z(l)i=W(l)h(l)i,(1)
		# self.fc = nn.Linear(in_dim, out_dim, bias=True)
		# equation (2)
		# e(l)ij=LeakyReLU(a⃗ (l)T(z(l)i|z(l)j)),(2)
		# self.attn_fc = nn.Linear(2 * out_dim, 1, bias=True)
		self.attn_fc = nn.Linear(2 * in_dim, 1, bias=True)

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
		# nn.init.xavier_uniform_(self.fc.weight)
		nn.init.xavier_uniform_(self.attn_fc.weight)

	def edge_attention(self, edges):
		# edge UDF for equation (2)
		obs_src_dest = torch.cat([edges.src['z_'], edges.dst['z_']], dim=1)
		a = self.attn_fc(obs_src_dest)
		# return {'e': F.leaky_relu(a)}
		return {'e': a}

	def message_func(self, edges):
		# message UDF for equation (3)
		# α(l)ij=exp(e(l)ij)∑k∈N(i)exp(e(l)ik),(3)
		return {'e': edges.data['e'], 'pi': edges.src['pi'], 'act': edges.src['act']}

	def reduce_func(self, nodes):
		# reduce UDF for equation (3)
		# equation (3)
		w = torch.sigmoid(nodes.mailbox['e'])
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
		# z_ = self.fc(h)
		# self.g.ndata['z_'] = z_
		self.g.ndata['z_'] = h
		self.g.ndata['pi'] = policies.reshape(-1,self.num_actions)
		self.g.ndata['act'] = actions.reshape(-1,self.num_actions)
		# equation (2)
		self.g.apply_edges(self.edge_attention)
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
		self.input_processor = GATLayerInput(preprocess_input_dim, preprocess_output_dim, num_agents)
		self.weight_layer = GATLayer(weight_input_dim, weight_output_dim, num_agents, num_actions)
		self.value_layer = ValueNetwork(input_dim, output_dim)

	def forward(self, g, policies, actions):
		# g = self.input_processor(g)
		# features = g.ndata['obs_proc']
		features = self.input_processor(g, g.ndata['obs'])
		g.ndata['obs_proc'] = features
		obs_final, weights = self.weight_layer(g,g.ndata['mypose_goalpose'],policies,actions)
		x = self.value_layer(obs_final)
		return x, weights


class PolicyNetwork(nn.Module):
	def __init__(
		self,
		policy_sizes
		):
		super(PolicyNetwork,self).__init__()

		self.policy = create_model(policy_sizes)

	def forward(self,states):
		return F.softmax(self.policy(states),-1)