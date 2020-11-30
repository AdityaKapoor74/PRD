from typing import Any, List, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

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
		merge_agent_inputs = F.relu((torch.sum(other_agent_outputs,dim=-2).unsqueeze(-2) + curr_agent_outputs)/self.num_agents)
		merge_agent_outputs = self.common(merge_agent_inputs)
		return merge_agent_outputs




class PolicyNetwork(nn.Module):
	def __init__(
		self,
		policy_sizes
		):
		super(PolicyNetwork,self).__init__()

		self.policy = create_model(policy_sizes)

	def forward(self,states):
		return F.softmax(self.policy(states),-1)