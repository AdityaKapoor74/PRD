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



class GATLayer(nn.Module):
	def __init__(self, g, in_dim, out_dim):
		super(GATLayer, self).__init__()
		self.g = g
		# equation (1)
		# z(l)i=W(l)h(l)i,(1)
		self.fc = nn.Linear(in_dim, out_dim, bias=False)
		# equation (2)
		# e(l)ij=LeakyReLU(a⃗ (l)T(z(l)i|z(l)j)),(2)
		self.attn_fc = nn.Linear(2 * out_dim, 1, bias=False)
		self.reset_parameters()

	def reset_parameters(self):
		"""Reinitialize learnable parameters."""
		gain = nn.init.calculate_gain('relu')
		nn.init.xavier_normal_(self.fc.weight, gain=gain)
		nn.init.xavier_normal_(self.attn_fc.weight, gain=gain)

	def edge_attention(self, edges):
		# edge UDF for equation (2)
		obs_src_dest = torch.cat([edges.src['obs'], edges.dst['obs']], dim=1)
		a = self.attn_fc(obs_src_dest)
		return {'e': F.leaky_relu(a)}

	def message_func(self, edges):
		# message UDF for equation (3)
		# α(l)ij=exp(e(l)ij)∑k∈N(i)exp(e(l)ik),(3)
		return {'z': edges.src['obs'], 'e': edges.data['e']}

	def reduce_func(self, nodes, action, policy):
		# reduce UDF for equation (3)
		# equation (3)
		alpha = F.softmax(nodes.mailbox['e'], dim=1)
		z = alpha*action + (1-alpha)*policy
		return {'z':z}

	def forward(self, h, action, policy):
		# equation (1)
		z = self.fc(h)
		self.g.ndata['z'] = z
		# equation (2)
		self.g.apply_edges(self.edge_attention)
		# equation (3) & (4)
		self.g.update_all(self.message_func, self.reduce_func)
		return self.g.ndata.pop('z')


class GNNLayer(nn.Module):
	def __init__(self, in_feats, out_feats):
		super(GNNLayer, self).__init__()
		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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



class CriticNetwork(nn.Module):
	def __init__(self, weight_input_dim, weight_output_dim, input_dim, output_dim):
		super(CriticNetwork, self).__init__()
		self.value_layer1 = GNNLayer(input_dim, 64)
		self.value_layer2 = GNNLayer(64, output_dim)

	def forward(self, g):
		features = g.ndata['obs']
		x = F.relu(self.value_layer1(g, features))
		x = self.value_layer2(g, x)
		return x


class PolicyNetwork(nn.Module):
	def __init__(
		self,
		policy_sizes
		):
		super(PolicyNetwork,self).__init__()

		self.policy = create_model(policy_sizes)

	def forward(self,states):
		return F.softmax(self.policy(states),-1)