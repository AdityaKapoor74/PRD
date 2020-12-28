from typing import Any, List, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl 
import numpy as np
import dgl
import dgl.function as fn
from dgl import DGLGraph

# *******************************************
# Q(s,a) 
# *******************************************

def create_model(
	layer_sizes: Tuple,
	weight_init: str = "xavier_uniform",
	activation_func: str = "relu"
	):

	layers = []
	limit = len(layer_sizes)

	# add more activations
	activation = nn.Tanh() if activation_func == "tanh" else nn.ReLU()

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



class GNNLayer(nn.Module):
	def __init__(self, in_feats, out_feats):
		super(GNNLayer, self).__init__()
		# self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		self.device = "cpu"
		self.linear = nn.Linear(in_feats, out_feats)
		torch.nn.init.xavier_uniform_(self.linear.weight)
		self.gcn_msg = fn.copy_src(src='obs', out='m')
		self.gcn_reduce = fn.mean(msg='m', out='obs')	

	def forward(self, g, feature):
		# Creating a local scope so that all the stored ndata and edata
		# (such as the `'h'` ndata below) are automatically popped out
		# when the scope exits.
		with g.local_scope():
			g.ndata['obs'] = feature
			g.update_all(self.gcn_msg, self.gcn_reduce)
			h = g.ndata['obs'].to(self.device)
			return self.linear(h)

class CriticInputPreprocess(nn.Module):
	def __init__(self, input_dim, output_dim):
		super(CriticInputPreprocess, self).__init__()
		self.value_layer1 = GNNLayer(input_dim, 64)
		self.value_layer2 = GNNLayer(64, output_dim)

	def forward(self, g):
		features = g.ndata['obs']
		x = F.relu(self.value_layer1(g, features))
		x = self.value_layer2(g, x)
		g.ndata['obs_proc'] = x
		return g


class GATLayer(nn.Module):
	def __init__(self, in_dim, out_dim, num_agents,num_actions):
		super(GATLayer, self).__init__()
		self.num_agents = num_agents
		self.num_actions = num_actions
		# self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		self.device = "cpu"
		# equation (1)
		# z(l)i=W(l)h(l)i,(1)
		self.fc = nn.Linear(in_dim, out_dim, bias=False)
		# equation (2)
		# e(l)ij=LeakyReLU(a⃗ (l)T(z(l)i|z(l)j)),(2)
		self.attn_fc = nn.Linear(2 * out_dim, 1, bias=False)

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
		gain = nn.init.calculate_gain('relu')
		nn.init.xavier_normal_(self.fc.weight, gain=gain)
		nn.init.xavier_normal_(self.attn_fc.weight, gain=gain)

	def edge_attention(self, edges):
		# edge UDF for equation (2)
		obs_src_dest = torch.cat([edges.src['z_'], edges.dst['z_']], dim=1)
		a = self.attn_fc(obs_src_dest)
		return {'e': F.leaky_relu(a)}

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
		z_ = self.fc(h)
		self.g.ndata['z_'] = z_
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
		self.input_processor = CriticInputPreprocess(preprocess_input_dim, preprocess_output_dim)
		self.weight_layer = GATLayer(weight_input_dim, weight_output_dim, num_agents, num_actions)
		self.value_layer = ValueNetwork(input_dim, output_dim)

	def forward(self, g, policies, actions):
		g = self.input_processor(g)
		features = g.ndata['obs_proc']
		obs_final, weights = self.weight_layer(g,features,policies,actions)
		x = self.value_layer(obs_final)
		return x, weights

# class PolicyNetwork(nn.Module):
# 	def __init__(self, input_dim, output_dim):
# 		super(PolicyNetwork,self).__init__()
# 		self.policy_layer1 = GNNLayer(input_dim, 64)
# 		self.policy_layer2 = GNNLayer(64, output_dim)

# 	def forward(self, g):
# 		features = g.ndata['obs']
# 		x = F.relu(self.policy_layer1(g, features))
# 		x = F.softmax(self.policy_layer2(g, x),-1)
# 		return x

class PolicyNetwork(nn.Module):
	def __init__(
		self,
		policy_sizes
		):
		super(PolicyNetwork,self).__init__()

		self.policy = create_model(policy_sizes)

	def forward(self,states):
		return F.softmax(self.policy(states),-1)