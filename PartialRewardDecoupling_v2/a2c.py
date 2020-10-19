from typing import Any, List, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

# *******************************************
# Q(s,a) 
# *******************************************
class GenericNetwork(nn.Module):
	def __init__(
	layer_sizes: Tuple,
	weight_init: str = "xavier_uniform",
	activation_func: str = "relu"
	):
		super(GenericNetwork,self).__init__()

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




class QValueNetwork(nn.Module):

	def __init__(self,curr_agent_sizes, other_agent_sizes, common_sizes):
		super(QValueNetwork,self).__init__()

		self.current_agent = GenericNetwork(curr_agent_sizes)
		self.other_agent = GenericNetwork(other_agent_sizes)
		self.common = GenericNetwork(common_sizes)




	def forward(current_agent_states, other_agent_states):

		curr_agent_outputs = torch.FloatTensor([[self.current_agent(torch.cat([current_agent[j],self.one_hots[i]])) for i in range(self.action_dim)] for j in range(self.num_agents)])
		other_agents_states_actions = torch.zeros(self.num_agents,self.num_agents-1,self.action_dim)
		# other_agent_outputs 



class PolicyNetwork(nn.Module):
	def __init__(policy_sizes):
		super(PolicyNetwork,self).__init__()

		self.policy = GenericNetwork(policy_sizes)

	def forward(states):
		return F.softmax(self.policy(states),-1)