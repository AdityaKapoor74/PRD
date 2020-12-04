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
	def __init__(self, input_dim, output_dim):
		super(CriticNetwork, self).__init__()
		self.layer1 = GNNLayer(input_dim, 64)
		self.layer2 = GNNLayer(64, output_dim)

	def forward(self, g):
		features = g.ndata['obs']
		x = F.relu(self.layer1(g, features))
		x = self.layer2(g, x)
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