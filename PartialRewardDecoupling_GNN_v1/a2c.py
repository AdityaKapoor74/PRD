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



class ValueNetwork(nn.Module):

	def __init__(self,
		curr_agent_sizes, 
		other_agent_sizes, 
		common_sizes,
		num_agents
		):
		super(ValueNetwork,self).__init__()

		self.current_agent = create_model(curr_agent_sizes)
		self.other_agent = create_model(other_agent_sizes)
		self.common = create_model(common_sizes)
		self.num_agents = num_agents



	def forward(self,current_agent_states, other_agent_states):
		curr_agent_outputs = self.current_agent(current_agent_states)
		other_agent_outputs = self.other_agent(other_agent_states)
		merge_agent_inputs = F.relu((torch.sum(other_agent_outputs,dim=2).unsqueeze(2) + curr_agent_outputs)/self.num_agents)
		merge_agent_outputs = self.common(merge_agent_inputs)
		return merge_agent_outputs


class GNNLayer(nn.Module):
	def __init__(self, in_feats, out_feats):
		super(GNNLayer, self).__init__()
		self.linear = nn.Linear(in_feats, out_feats)
		self.gcn_msg = fn.copy_src(src='h', out='m')
		self.gcn_reduce = fn.mean(msg='m', out='h')	

	def forward(self, g, feature):
		# Creating a local scope so that all the stored ndata and edata
		# (such as the `'h'` ndata below) are automatically popped out
		# when the scope exits.
		with g.local_scope():
			g.ndata['h'] = feature
			g.update_all(self.gcn_msg, self.gcn_reduce)
			h = g.ndata['h']
			return self.linear(h)


class CriticNetwork(nn.Module):
	def __init__(self, input_dim, hidden_dim, output_dim):
		super(CriticNetwork, self).__init__()
		self.layer1 = GNNLayer(input_dim, hidden_dim)
		torch.nn.init.xavier_uniform_(self.layer1.weight)
		self.layer2 = GNNLayer(hidden_dim, output_dim)
		torch.nn.init.xavier_uniform_(self.layer2.weight)

	def forward(self, g, features):
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