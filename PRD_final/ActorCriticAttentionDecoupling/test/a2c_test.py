from typing import Any, List, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import datetime
import math
from itertools import chain


class StateActionGATCritic(nn.Module):
	def __init__(self, obs_input_dim, obs_output_dim, obs_act_input_dim, obs_act_output_dim, final_input_dim, final_output_dim, num_agents, num_actions, threshold=0.1):
		super(StateActionGATCritic, self).__init__()
		
		self.num_agents = num_agents
		self.num_actions = num_actions
		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

		self.key_layer = nn.Linear(obs_input_dim, obs_output_dim, bias=False)
		self.query_layer = nn.Linear(obs_input_dim, obs_output_dim, bias=False)
		self.attention_value_layer = nn.Linear(obs_act_input_dim, obs_act_output_dim, bias=False)
		# dimesion of key
		self.d_k_obs_act = obs_output_dim

		# NOISE
		self.noise_normal = torch.distributions.Normal(loc=torch.tensor([0.0]), scale=torch.tensor([1.0]))
		self.noise_uniform = torch.rand
		# ********************************************************************************************************

		# ********************************************************************************************************
		# FCN FINAL LAYER TO GET VALUES
		self.final_value_layer_1 = nn.Linear(final_input_dim, 64, bias=False)
		self.final_value_layer_2 = nn.Linear(64, final_output_dim, bias=False)
		# ********************************************************************************************************	

		self.place_policies = torch.zeros(self.num_agents,self.num_agents,obs_act_input_dim).to(self.device)
		self.place_actions = torch.ones(self.num_agents,self.num_agents,obs_act_input_dim).to(self.device)
		one_hots = torch.ones(obs_act_input_dim)
		zero_hots = torch.zeros(obs_act_input_dim)

		for j in range(self.num_agents):
			self.place_policies[j][j] = one_hots
			self.place_actions[j][j] = zero_hots

		self.threshold = threshold
		self.obs_act_input_dim = obs_act_input_dim
		# ********************************************************************************************************* 

		self.reset_parameters()


	def reset_parameters(self):
		"""Reinitialize learnable parameters."""
		gain_leaky = nn.init.calculate_gain('leaky_relu')

		nn.init.xavier_uniform_(self.key_layer.weight)
		nn.init.xavier_uniform_(self.query_layer.weight)
		nn.init.xavier_uniform_(self.attention_value_layer.weight)


		nn.init.xavier_uniform_(self.final_value_layer_1.weight, gain=gain_leaky)
		nn.init.xavier_uniform_(self.final_value_layer_2.weight, gain=gain_leaky)



	def forward(self, states, policies, actions):

		# input to KEY, QUERY and ATTENTION VALUE NETWORK
		obs_actions = torch.cat([states,actions],dim=-1)
		# For calculating the right advantages
		obs_policy = torch.cat([states,policies], dim=-1)

		# KEYS
		key_obs_actions = self.key_layer(states)
		# QUERIES
		query_obs_actions = self.query_layer(states)
		# SCORE
		score_obs_actions = torch.bmm(query_obs_actions,key_obs_actions.transpose(1,2)).reshape(-1,1)
		score_obs_actions = score_obs_actions.reshape(-1,self.num_agents,1)
		# WEIGHT
		weight = F.softmax(score_obs_actions/math.sqrt(self.d_k_obs_act), dim=-2)
		weight = weight.reshape(weight.shape[0]//self.num_agents,self.num_agents,-1)
		ret_weight = weight
		
		obs_actions = obs_actions.repeat(1,self.num_agents,1).reshape(obs_actions.shape[0],self.num_agents,self.num_agents,-1)
		obs_policy = obs_policy.repeat(1,self.num_agents,1).reshape(obs_policy.shape[0],self.num_agents,self.num_agents,-1)
		obs_actions_policies = self.place_policies*obs_policy + self.place_actions*obs_actions
		attention_values = self.attention_value_layer(obs_actions_policies)
		attention_values = attention_values.repeat(1,self.num_agents,1,1).reshape(attention_values.shape[0],self.num_agents,self.num_agents,self.num_agents,-1)
		
		weight = weight.unsqueeze(-2).repeat(1,1,self.num_agents,1).unsqueeze(-1)
		weighted_attention_values = attention_values*weight

		node_features = torch.sum(weighted_attention_values, dim=-2)

		Value = F.leaky_relu(self.final_value_layer_1(node_features))
		Value = self.final_value_layer_2(Value)

		return Value, ret_weight


class StateOnlyGATCritic(nn.Module):
	def __init__(self, obs_input_dim, obs_output_dim, final_input_dim, final_output_dim, num_agents, num_actions, threshold=0.1):
		super(StateOnlyGATCritic, self).__init__()
		
		self.num_agents = num_agents
		self.num_actions = num_actions
		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

		self.key_layer = nn.Linear(obs_input_dim, obs_output_dim, bias=False)
		self.query_layer = nn.Linear(obs_input_dim, obs_output_dim, bias=False)
		self.attention_value_layer = nn.Linear(obs_input_dim, obs_output_dim, bias=False)
		# dimesion of key
		self.d_k_obs_act = obs_output_dim

		# NOISE
		self.noise_normal = torch.distributions.Normal(loc=torch.tensor([0.0]), scale=torch.tensor([1.0]))
		self.noise_uniform = torch.rand
		# ********************************************************************************************************

		# ********************************************************************************************************
		# FCN FINAL LAYER TO GET VALUES
		self.final_value_layer_1 = nn.Linear(final_input_dim, 64, bias=False)
		self.final_value_layer_2 = nn.Linear(64, final_output_dim, bias=False)
		# ********************************************************************************************************* 

		self.reset_parameters()


	def reset_parameters(self):
		"""Reinitialize learnable parameters."""
		gain_leaky = nn.init.calculate_gain('leaky_relu')

		nn.init.xavier_uniform_(self.key_layer.weight)
		nn.init.xavier_uniform_(self.query_layer.weight)
		nn.init.xavier_uniform_(self.attention_value_layer.weight)

		nn.init.xavier_uniform_(self.final_value_layer_1.weight, gain=gain_leaky)
		nn.init.xavier_uniform_(self.final_value_layer_2.weight, gain=gain_leaky)



	def forward(self, states, policies=None, actions=None):
		# KEYS
		key_obs_actions = self.key_layer(states)
		# QUERIES
		query_obs_actions = self.query_layer(states)
		# SCORE
		score_obs_actions = torch.bmm(query_obs_actions,key_obs_actions.transpose(1,2)).reshape(-1,1)
		score_obs_actions = score_obs_actions.reshape(-1,self.num_agents,1)
		# WEIGHT
		weight = F.softmax(score_obs_actions/math.sqrt(self.d_k_obs_act), dim=-2)
		weight = weight.reshape(weight.shape[0]//self.num_agents,self.num_agents,-1)
		# ATTENTION VALUES
		attention_values = self.attention_value_layer(states)
		# AGGREGATION
		node_features = torch.matmul(weight,attention_values)

		Value = F.leaky_relu(self.final_value_layer_1(node_features))
		Value = self.final_value_layer_2(Value)

		Value = torch.stack(self.num_agents*[Value],2)

		return Value, weight
		


class StateOnlyMLPCritic(nn.Module):
	def __init__(self,state_dim,num_agents):
		super(StateOnlyMLPCritic,self).__init__()

		self.state_dim = state_dim
		self.num_agents = num_agents		
		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


		self.fc1 = nn.Linear(state_dim*num_agents,128)
		self.fc2 = nn.Linear(128,128)
		self.fc3 = nn.Linear(128,1)

	def forward(self, states, policies=None, actions=None):

		# T x num_agents x state_dim
		T = states.shape[0]
		
		states_aug = [torch.roll(states,i,1) for i in range(self.num_agents)]

		states_aug = torch.cat(states_aug,dim=2)

		x = self.fc1(states_aug)
		x = nn.ReLU()(x)
		x = self.fc2(x)
		x = nn.ReLU()(x)
		V = self.fc3(x)

		V = torch.stack(self.num_agents*[V],2)

		return V, 1/self.num_agents*torch.ones((T,self.num_agents,self.num_agents),device=self.device)



class StateActionMLPCritic(nn.Module):
	def __init__(self,state_dim,action_dim,num_agents):
		super(StateActionMLPCritic,self).__init__()

		self.state_dim = state_dim
		self.num_agents = num_agents		
		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


		self.fc1 = nn.Linear((state_dim+action_dim)*num_agents,128)
		self.fc2 = nn.Linear(128,128)
		self.fc3 = nn.Linear(128,1)


		self.place_policies = torch.zeros(self.num_agents,self.num_agents,state_dim+action_dim).to(self.device)
		self.place_actions = torch.ones(self.num_agents,self.num_agents,state_dim+action_dim).to(self.device)
		one_hots = torch.ones(state_dim+action_dim)
		zero_hots = torch.zeros(state_dim+action_dim)

		for j in range(self.num_agents):
			self.place_policies[j][j] = one_hots
			self.place_actions[j][j] = zero_hots

	def forward(self, states, policies, actions):

		# T x num_agents x state_dim
		T = states.shape[0]

		states_actions = torch.cat([states, actions], dim=-1)
		states_policies = torch.cat([states, policies], dim=-1)
		states_actions = states_actions.repeat(1,self.num_agents,1).reshape(states_actions.shape[0],self.num_agents,self.num_agents,-1)
		states_policies = states_policies.repeat(1,self.num_agents,1).reshape(states_policies.shape[0],self.num_agents,self.num_agents,-1)
		states_actions_policies = self.place_policies*states_policies + self.place_actions*states_actions
		states_actions_policies = states_actions_policies.reshape(states_actions_policies.shape[0],self.num_agents,-1)

		x = self.fc1(states_actions_policies)
		x = nn.ReLU()(x)
		x = self.fc2(x)
		x = nn.ReLU()(x)
		V = self.fc3(x)

		V = torch.stack(self.num_agents*[V],2)

		return V, 1/self.num_agents*torch.ones((T,self.num_agents,self.num_agents),device=self.device)



class MLPPolicyNetwork(nn.Module):
	def __init__(self,state_dim,num_agents,action_dim):
		super(MLPPolicyNetwork,self).__init__()

		self.state_dim = state_dim
		self.num_agents = num_agents		
		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


		self.fc1 = nn.Linear(state_dim*num_agents,128)
		self.fc2 = nn.Linear(128,128)
		self.fc3 = nn.Linear(128,action_dim)

	def forward(self, states):

		# T x num_agents x state_dim
		T = states.shape[0]
		
		# [s0;s1;s2;s3]  -> [s0 s1 s2 s3; s1 s2 s3 s0; s2 s3 s1 s0 ....]

		states_aug = [torch.roll(states,i,1) for i in range(self.num_agents)]

		states_aug = torch.cat(states_aug,dim=2)

		x = self.fc1(states_aug)
		x = nn.ReLU()(x)
		x = self.fc2(x)
		x = nn.ReLU()(x)
		x = self.fc3(x)

		Policy = F.softmax(x, dim=-1)

		return Policy, 1/self.num_agents*torch.ones((T,self.num_agents,self.num_agents),device=self.device)


# class AttentionCritic(nn.Module):
# 	"""
# 	Attention network, used as critic for all agents. Each agent gets its own
# 	observation and action, and can also attend over the other agents' encoded
# 	observations and actions.
# 	"""
# 	def __init__(self, sa_sizes, hidden_dim=32, norm_in=True, attend_heads=1):
# 		"""
# 		Inputs:
# 			sa_sizes (list of (int, int)): Size of state and action spaces per
# 										  agent
# 			hidden_dim (int): Number of hidden dimensions
# 			norm_in (bool): Whether to apply BatchNorm to input
# 			attend_heads (int): Number of attention heads to use (use a number
# 								that hidden_dim is divisible by)
# 		"""
# 		super(AttentionCritic, self).__init__()
# 		assert (hidden_dim % attend_heads) == 0
# 		self.sa_sizes = sa_sizes
# 		self.nagents = len(sa_sizes)
# 		self.attend_heads = attend_heads

# 		self.critic_encoders = nn.ModuleList()
# 		self.critics = nn.ModuleList()

# 		self.state_encoders = nn.ModuleList()
# 		# iterate over agents
# 		for sdim, adim in sa_sizes:
# 			idim = sdim + adim
# 			odim = adim
# 			encoder = nn.Sequential()
# 			if norm_in:
# 				encoder.add_module('enc_bn', nn.BatchNorm1d(idim,
# 															affine=False))
# 			encoder.add_module('enc_fc1', nn.Linear(idim, hidden_dim))
# 			encoder.add_module('enc_nl', nn.LeakyReLU())
# 			self.critic_encoders.append(encoder)
# 			critic = nn.Sequential()
# 			critic.add_module('critic_fc1', nn.Linear(2 * hidden_dim,
# 													  hidden_dim))
# 			critic.add_module('critic_nl', nn.LeakyReLU())
# 			critic.add_module('critic_fc2', nn.Linear(hidden_dim, odim))
# 			self.critics.append(critic)

# 			state_encoder = nn.Sequential()
# 			if norm_in:
# 				state_encoder.add_module('s_enc_bn', nn.BatchNorm1d(
# 											sdim, affine=False))
# 			state_encoder.add_module('s_enc_fc1', nn.Linear(sdim,
# 															hidden_dim))
# 			state_encoder.add_module('s_enc_nl', nn.LeakyReLU())
# 			self.state_encoders.append(state_encoder)

# 		attend_dim = hidden_dim // attend_heads
# 		self.key_extractors = nn.ModuleList()
# 		self.selector_extractors = nn.ModuleList()
# 		self.value_extractors = nn.ModuleList()
# 		for i in range(attend_heads):
# 			self.key_extractors.append(nn.Linear(hidden_dim, attend_dim, bias=False))
# 			self.selector_extractors.append(nn.Linear(hidden_dim, attend_dim, bias=False))
# 			self.value_extractors.append(nn.Sequential(nn.Linear(hidden_dim,
# 																attend_dim),
# 													   nn.LeakyReLU()))

# 		self.shared_modules = [self.key_extractors, self.selector_extractors,
# 							   self.value_extractors, self.critic_encoders]

# 	def shared_parameters(self):
# 		"""
# 		Parameters shared across agents and reward heads
# 		"""
# 		return chain(*[m.parameters() for m in self.shared_modules])

# 	def scale_shared_grads(self):
# 		"""
# 		Scale gradients for parameters that are shared since they accumulate
# 		gradients from the critic loss function multiple times
# 		"""
# 		for p in self.shared_parameters():
# 			p.grad.data.mul_(1. / self.nagents)

# 	def forward(self, inps, agents=None, return_q=True, return_all_q=False,
# 				regularize=False, return_attend=False, logger=None, niter=0):
# 		"""
# 		Inputs:
# 			inps (list of PyTorch Matrices): Inputs to each agents' encoder
# 											 (batch of obs + ac)
# 			agents (int): indices of agents to return Q for
# 			return_q (bool): return Q-value
# 			return_all_q (bool): return Q-value for all actions
# 			regularize (bool): returns values to add to loss function for
# 							   regularization
# 			return_attend (bool): return attention weights per agent
# 			logger (TensorboardX SummaryWriter): If passed in, important values
# 												 are logged
# 		"""
# 		if agents is None:
# 			agents = range(len(self.critic_encoders))
# 		states = [s for s, a in inps]
# 		actions = [a for s, a in inps]
# 		inps = [torch.cat((s, a), dim=1) for s, a in inps]
# 		# extract state-action encoding for each agent
# 		sa_encodings = [encoder(inp) for encoder, inp in zip(self.critic_encoders, inps)]
# 		# extract state encoding for each agent that we're returning Q for
# 		s_encodings = [self.state_encoders[a_i](states[a_i]) for a_i in agents]
# 		# extract keys for each head for each agent
# 		all_head_keys = [[k_ext(enc) for enc in sa_encodings] for k_ext in self.key_extractors]
# 		# extract sa values for each head for each agent
# 		all_head_values = [[v_ext(enc) for enc in sa_encodings] for v_ext in self.value_extractors]
# 		# extract selectors for each head for each agent that we're returning Q for
# 		all_head_selectors = [[sel_ext(enc) for i, enc in enumerate(s_encodings) if i in agents]
# 							  for sel_ext in self.selector_extractors]

# 		other_all_values = [[] for _ in range(len(agents))]
# 		all_attend_logits = [[] for _ in range(len(agents))]
# 		all_attend_probs = [[] for _ in range(len(agents))]
# 		# calculate attention per head
# 		for curr_head_keys, curr_head_values, curr_head_selectors in zip(
# 				all_head_keys, all_head_values, all_head_selectors):
# 			# iterate over agents
# 			for i, a_i, selector in zip(range(len(agents)), agents, curr_head_selectors):
# 				keys = [k for j, k in enumerate(curr_head_keys) if j != a_i]
# 				values = [v for j, v in enumerate(curr_head_values) if j != a_i]
# 				# calculate attention across agents
# 				attend_logits = torch.matmul(selector.view(selector.shape[0], 1, -1),
# 											 torch.stack(keys).permute(1, 2, 0))
# 				# scale dot-products by size of key (from Attention is All You Need)
# 				scaled_attend_logits = attend_logits / np.sqrt(keys[0].shape[1])
# 				attend_weights = F.softmax(scaled_attend_logits, dim=2)
# 				other_values = (torch.stack(values).permute(1, 2, 0) *
# 								attend_weights).sum(dim=2)
# 				other_all_values[i].append(other_values)
# 				all_attend_logits[i].append(attend_logits)
# 				all_attend_probs[i].append(attend_weights)
# 		# calculate Q per agent
# 		all_rets = []
# 		for i, a_i in enumerate(agents):
# 			head_entropies = [(-((probs + 1e-8).log() * probs).squeeze().sum(1)
# 							   .mean()) for probs in all_attend_probs[i]]
# 			agent_rets = []
# 			critic_in = torch.cat((s_encodings[i], *other_all_values[i]), dim=1)
# 			all_q = self.critics[a_i](critic_in)
# 			int_acs = actions[a_i].max(dim=1, keepdim=True)[1]
# 			q = all_q.gather(1, int_acs)
# 			if return_q:
# 				agent_rets.append(q)
# 			if return_all_q:
# 				agent_rets.append(all_q)
# 			if regularize:
# 				# regularize magnitude of attention logits
# 				attend_mag_reg = 1e-3 * sum((logit**2).mean() for logit in
# 											all_attend_logits[i])
# 				regs = (attend_mag_reg,)
# 				agent_rets.append(regs)
# 			if return_attend:
# 				agent_rets.append(np.array(all_attend_probs[i]))
# 			if logger is not None:
# 				logger.add_scalars('agent%i/attention' % a_i,
# 								   dict(('head%i_entropy' % h_i, ent) for h_i, ent
# 										in enumerate(head_entropies)),
# 								   niter)
# 			if len(agent_rets) == 1:
# 				all_rets.append(agent_rets[0])
# 			else:
# 				all_rets.append(agent_rets)
# 		if len(all_rets) == 1:
# 			return all_rets[0]
# 		else:
# 			return all_rets

'''
Attention Actor Critic --> MAAC
'''
class AttentionCriticV1(nn.Module):
	def __init__(self, obs_input_dim, obs_output_dim, obs_act_input_dim, obs_act_output_dim, final_input_dim, final_output_dim, num_agents, num_actions, threshold=0.1, attend_heads=1):
		super(AttentionCriticV1, self).__init__()
		self.hidden_dim = 128
		assert (self.hidden_dim % attend_heads) == 0
		self.num_agents = num_agents
		self.attend_heads = attend_heads
		self.num_actions = num_actions
		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

		self.critic_encoders = nn.ModuleList().to(self.device)
		self.critics = nn.ModuleList().to(self.device)

		self.state_encoders = nn.ModuleList().to(self.device)
		# iterate over agents
		for i in range(self.num_agents):
			sdim = obs_input_dim
			idim = obs_act_input_dim
			odim = final_output_dim
			encoder = nn.Sequential()
			encoder.add_module('enc_fc1', nn.Linear(idim, self.hidden_dim))
			encoder.add_module('enc_nl', nn.LeakyReLU())
			self.critic_encoders.append(encoder)
			critic = nn.Sequential()
			critic.add_module('critic_fc1', nn.Linear(self.hidden_dim,
													  self.hidden_dim))
			critic.add_module('critic_nl', nn.LeakyReLU())
			critic.add_module('critic_fc2', nn.Linear(self.hidden_dim, odim))
			self.critics.append(critic)

			state_encoder = nn.Sequential()
			state_encoder.add_module('s_enc_fc1', nn.Linear(sdim,
															self.hidden_dim))
			state_encoder.add_module('s_enc_nl', nn.LeakyReLU())
			self.state_encoders.append(state_encoder)

		self.attend_dim = self.hidden_dim // self.attend_heads
		self.key_extractors = nn.ModuleList().to(self.device)
		self.selector_extractors = nn.ModuleList().to(self.device)
		self.value_extractors = nn.ModuleList().to(self.device)
		for i in range(attend_heads):
			self.key_extractors.append(nn.Linear(self.hidden_dim, self.attend_dim, bias=False))
			self.selector_extractors.append(nn.Linear(self.hidden_dim, self.attend_dim, bias=False))
			self.value_extractors.append(nn.Sequential(nn.Linear(self.hidden_dim,
																self.attend_dim),
													   nn.LeakyReLU()))

		self.shared_modules = [self.key_extractors, self.selector_extractors,
							   self.value_extractors, self.critic_encoders]


		self.place_policies = torch.zeros(self.num_agents,self.num_agents,obs_act_input_dim).to(self.device)
		self.place_actions = torch.ones(self.num_agents,self.num_agents,obs_act_input_dim).to(self.device)
		one_hots = torch.ones(obs_act_input_dim)
		zero_hots = torch.zeros(obs_act_input_dim)

		for j in range(self.num_agents):
			self.place_policies[j][j] = one_hots
			self.place_actions[j][j] = zero_hots

		self.threshold = threshold

	def shared_parameters(self):
		"""
		Parameters shared across agents and reward heads
		"""
		return chain(*[m.parameters() for m in self.shared_modules])

	def scale_shared_grads(self):
		"""
		Scale gradients for parameters that are shared since they accumulate
		gradients from the critic loss function multiple times
		"""
		for p in self.shared_parameters():
			p.grad.data.mul_(1. / self.num_agents)


	def forward(self,states, actions, policies):
		obs_actions = torch.cat([states,actions],dim=-1)
		obs_policy = torch.cat([states,policies], dim=-1)
		obs_actions = obs_actions.repeat(1,self.num_agents,1).reshape(obs_actions.shape[0],self.num_agents,self.num_agents,-1)
		obs_policy = obs_policy.repeat(1,self.num_agents,1).reshape(obs_policy.shape[0],self.num_agents,self.num_agents,-1)
		obs_actions_policies = self.place_policies*obs_policy + self.place_actions*obs_actions

		state_encodings = torch.zeros(states.shape[0],self.num_agents,self.hidden_dim).to(self.device)
		state_action_policy_encodings = torch.zeros(obs_actions_policies.shape[0],self.num_agents,self.num_agents,self.hidden_dim).to(self.device)

		for i in range(self.num_agents):
			state_encodings[:,i,:] = self.state_encoders[i](states[:,i,:])
			state_action_policy_encodings[:,i,:,:] = self.critic_encoders[i](obs_actions_policies[:,i,:,:])

		weights_attn = []
		node_features = []
		for i in range(self.attend_heads):
			key = self.key_extractors[i](state_encodings)
			query = self.selector_extractors[i](state_encodings)
			weight = F.softmax(torch.matmul(query,key.transpose(1,2))/math.sqrt(self.attend_dim),dim=-1)
			weights_attn.append(weight)

			attention_values = self.value_extractors[i](state_action_policy_encodings)
			attention_values = attention_values.repeat(1,self.num_agents,1,1).reshape(attention_values.shape[0],self.num_agents,self.num_agents,self.num_agents,-1)
			
			weight = weight.unsqueeze(-2).repeat(1,1,self.num_agents,1).unsqueeze(-1)
			weighted_attention_values = attention_values*weight
			node_features.append(torch.sum(weighted_attention_values, dim=-2))

		multi_head_node_features = torch.cat([feature for feature in node_features], dim=-1).to(self.device)

		Values = torch.zeros(multi_head_node_features.shape[0],self.num_agents,self.num_agents,1).to(self.device)
		for i in range(self.num_agents):
			Values[:,i,:,:] = self.critics[i](multi_head_node_features[:,i,:,:])

		return Values, weights_attn



class StateActionGATCriticMultiHead(nn.Module):
	def __init__(self, obs_input_dim, obs_output_dim, obs_act_input_dim, obs_act_output_dim, final_input_dim, final_output_dim, num_agents, num_actions, threshold=0.1, attention_heads=4):
		super(StateActionGATCriticMultiHead, self).__init__()
		assert obs_act_output_dim%attention_heads == 0
		assert final_input_dim == attention_heads*obs_act_output_dim

		self.num_agents = num_agents
		self.num_actions = num_actions
		self.attention_heads = attention_heads
		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

		self.state_embed = nn.Sequential(nn.Linear(obs_input_dim, 128), nn.LeakyReLU())
		self.state_act_pol_embed = nn.Sequential(nn.Linear(obs_act_input_dim, 128), nn.LeakyReLU())

		self.key_layers = nn.ModuleList().to(self.device)
		self.query_layers = nn.ModuleList().to(self.device)
		self.attention_value_layers = nn.ModuleList().to(self.device)
		self.layer_norms = nn.ModuleList().to(self.device)

		for i in range(self.attention_heads):
			self.key_layers.append(nn.Linear(128, obs_output_dim, bias=False))
			self.query_layers.append(nn.Linear(128, obs_output_dim, bias=False))
			self.attention_value_layers.append(nn.Linear(128, obs_act_output_dim, bias=False))
			self.layer_norms.append(nn.LayerNorm(obs_act_output_dim))

		# dimesion of key
		self.d_k_obs_act = obs_output_dim

		# NOISE
		self.noise_normal = torch.distributions.Normal(loc=torch.tensor([0.0]), scale=torch.tensor([1.0]))
		self.noise_uniform = torch.rand
		# ********************************************************************************************************

		# ********************************************************************************************************
		# FCN FINAL LAYER TO GET VALUES
		self.final_value_layer_1 = nn.Linear(final_input_dim, 64, bias=False)
		self.final_value_layer_2 = nn.Linear(64, final_output_dim, bias=False)
		# ********************************************************************************************************	

		self.place_policies = torch.zeros(self.num_agents,self.num_agents,obs_act_input_dim).to(self.device)
		self.place_actions = torch.ones(self.num_agents,self.num_agents,obs_act_input_dim).to(self.device)
		one_hots = torch.ones(obs_act_input_dim)
		zero_hots = torch.zeros(obs_act_input_dim)

		for j in range(self.num_agents):
			self.place_policies[j][j] = one_hots
			self.place_actions[j][j] = zero_hots

		self.threshold = threshold
		self.obs_act_input_dim = obs_act_input_dim
		# ********************************************************************************************************* 

		self.shared_modules = [self.state_embed[0], self.key_layers, self.query_layers, self.state_act_pol_embed[0], self.attention_value_layers, self.final_value_layer_1, self.final_value_layer_2]

		self.reset_parameters()


	def reset_parameters(self):
		"""Reinitialize learnable parameters."""
		gain_leaky = nn.init.calculate_gain('leaky_relu')

		nn.init.xavier_uniform_(self.state_embed[0].weight, gain=gain_leaky)
		nn.init.xavier_uniform_(self.state_act_pol_embed[0].weight, gain=gain_leaky)

		for i in range(self.attention_heads):
			nn.init.xavier_uniform_(self.key_layers[i].weight)
			nn.init.xavier_uniform_(self.query_layers[i].weight)
			nn.init.xavier_uniform_(self.attention_value_layers[i].weight)


		nn.init.xavier_uniform_(self.final_value_layer_1.weight, gain=gain_leaky)
		nn.init.xavier_uniform_(self.final_value_layer_2.weight, gain=gain_leaky)


	def shared_parameters(self):
		"""
		Parameters shared across agents and reward heads
		"""
		return chain(*[m.parameters() for m in self.shared_modules])

	def scale_shared_grads(self):
		"""
		Scale gradients for parameters that are shared since they accumulate
		gradients from the critic loss function multiple times
		"""
		for p in self.shared_parameters():
			p.grad.data.mul_(1. / self.num_agents)



	def forward(self, states, policies, actions):
		# EMBED STATES
		states_embed = self.state_embed(states)

		obs_actions = torch.cat([states,actions],dim=-1)
		obs_policy = torch.cat([states,policies], dim=-1)
		obs_actions = obs_actions.repeat(1,self.num_agents,1).reshape(obs_actions.shape[0],self.num_agents,self.num_agents,-1)
		obs_policy = obs_policy.repeat(1,self.num_agents,1).reshape(obs_policy.shape[0],self.num_agents,self.num_agents,-1)
		obs_actions_policies = self.place_policies*obs_policy + self.place_actions*obs_actions
		# EMBED STATE ACTION POLICY
		obs_actions_policies_embed = self.state_act_pol_embed(obs_actions_policies)

		weights_attn = []
		node_features = []

		for i in range(self.attention_heads):
			key = self.key_layers[i](states_embed)
			query = self.query_layers[i](states_embed)
			weight = F.softmax(torch.matmul(query,key.transpose(1,2))/math.sqrt(self.d_k_obs_act),dim=-1)
			weights_attn.append(weight)

			attention_values = self.attention_value_layers[i](obs_actions_policies_embed)
			attention_values = attention_values.repeat(1,self.num_agents,1,1).reshape(attention_values.shape[0],self.num_agents,self.num_agents,self.num_agents,-1)
			
			weight = weight.unsqueeze(-2).repeat(1,1,self.num_agents,1).unsqueeze(-1)
			weighted_attention_values = attention_values*weight
			node_features.append(self.layer_norms[i](torch.sum(weighted_attention_values, dim=-2)))

		multi_head_node_features = torch.cat([feature for feature in node_features], dim=-1).to(self.device)

		Value = F.leaky_relu(self.final_value_layer_1(multi_head_node_features))
		Value = self.final_value_layer_2(Value)

		return Value, weights_attn




'''
V1
'''
class StateActionGATCriticWoResConnV1(nn.Module):
	def __init__(self, obs_input_dim, obs_output_dim, obs_act_input_dim, obs_act_output_dim, final_input_dim, final_output_dim, num_agents, num_actions, threshold=0.1):
		super(StateActionGATCriticWoResConnV1, self).__init__()
		
		self.num_agents = num_agents
		self.num_actions = num_actions
		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

		self.state_embed = nn.Sequential(nn.Linear(obs_input_dim, 128), nn.LeakyReLU())
		self.key_layer = nn.Linear(128, obs_output_dim, bias=False)
		self.query_layer = nn.Linear(128, obs_output_dim, bias=False)
		self.state_act_pol_embed = nn.Sequential(nn.Linear(obs_act_input_dim, 128), nn.LeakyReLU())
		self.attention_value_layer = nn.Linear(128, obs_act_output_dim, bias=False)
		# dimesion of key
		self.d_k_obs_act = obs_output_dim

		# NOISE
		self.noise_normal = torch.distributions.Normal(loc=torch.tensor([0.0]), scale=torch.tensor([1.0]))
		self.noise_uniform = torch.rand
		# ********************************************************************************************************

		# ********************************************************************************************************
		# FCN FINAL LAYER TO GET VALUES
		self.final_value_layer_1 = nn.Linear(final_input_dim, 64, bias=False)
		self.final_value_layer_2 = nn.Linear(64, final_output_dim, bias=False)
		# ********************************************************************************************************	

		self.place_policies = torch.zeros(self.num_agents,self.num_agents,obs_act_input_dim).to(self.device)
		self.place_actions = torch.ones(self.num_agents,self.num_agents,obs_act_input_dim).to(self.device)
		one_hots = torch.ones(obs_act_input_dim)
		zero_hots = torch.zeros(obs_act_input_dim)

		for j in range(self.num_agents):
			self.place_policies[j][j] = one_hots
			self.place_actions[j][j] = zero_hots

		self.threshold = threshold
		# ********************************************************************************************************* 

		self.reset_parameters()


	def reset_parameters(self):
		"""Reinitialize learnable parameters."""
		gain_leaky = nn.init.calculate_gain('leaky_relu')

		nn.init.xavier_uniform_(self.state_embed[0].weight, gain=gain_leaky)
		nn.init.xavier_uniform_(self.state_act_pol_embed[0].weight, gain=gain_leaky)

		nn.init.xavier_uniform_(self.key_layer.weight)
		nn.init.xavier_uniform_(self.query_layer.weight)
		nn.init.xavier_uniform_(self.attention_value_layer.weight)


		nn.init.xavier_uniform_(self.final_value_layer_1.weight, gain=gain_leaky)
		nn.init.xavier_uniform_(self.final_value_layer_2.weight, gain=gain_leaky)



	def forward(self, states, policies, actions):
		# EMBED STATES
		states_embed = self.state_embed(states)
		# KEYS
		key_obs = self.key_layer(states_embed)
		# QUERIES
		query_obs = self.query_layer(states_embed)
		# WEIGHT
		weight = F.softmax(torch.matmul(query_obs,key_obs.transpose(1,2))/math.sqrt(self.d_k_obs_act),dim=-1)
		ret_weight = weight

		obs_actions = torch.cat([states,actions],dim=-1)
		obs_policy = torch.cat([states,policies], dim=-1)
		obs_actions = obs_actions.repeat(1,self.num_agents,1).reshape(obs_actions.shape[0],self.num_agents,self.num_agents,-1)
		obs_policy = obs_policy.repeat(1,self.num_agents,1).reshape(obs_policy.shape[0],self.num_agents,self.num_agents,-1)
		obs_actions_policies = self.place_policies*obs_policy + self.place_actions*obs_actions
		# EMBED STATE ACTION POLICY
		obs_actions_policies_embed = self.state_act_pol_embed(obs_actions_policies)
		attention_values = self.attention_value_layer(obs_actions_policies_embed)
		attention_values = attention_values.repeat(1,self.num_agents,1,1).reshape(attention_values.shape[0],self.num_agents,self.num_agents,self.num_agents,-1)
		
		weight = weight.unsqueeze(-2).repeat(1,1,self.num_agents,1).unsqueeze(-1)
		weighted_attention_values = attention_values*weight
		node_features = torch.sum(weighted_attention_values, dim=-2)

		Value = F.leaky_relu(self.final_value_layer_1(node_features))
		Value = self.final_value_layer_2(Value)

		return Value, ret_weight



class StateActionGATCriticWResConnV1(nn.Module):
	def __init__(self, obs_input_dim, obs_output_dim, obs_act_input_dim, obs_act_output_dim, final_input_dim, final_output_dim, num_agents, num_actions, threshold=0.1):
		super(StateActionGATCriticWResConnV1, self).__init__()
		
		self.num_agents = num_agents
		self.num_actions = num_actions
		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

		self.state_embed = nn.Sequential(nn.Linear(obs_input_dim, 128), nn.LeakyReLU())
		self.key_layer = nn.Linear(128, obs_output_dim, bias=False)
		self.query_layer = nn.Linear(128, obs_output_dim, bias=False)
		self.state_act_pol_embed = nn.Sequential(nn.Linear(obs_act_input_dim, 128), nn.LeakyReLU())
		self.attention_value_layer = nn.Linear(128, obs_act_output_dim, bias=False)
		# dimesion of key
		self.d_k_obs_act = obs_output_dim

		# NOISE
		self.noise_normal = torch.distributions.Normal(loc=torch.tensor([0.0]), scale=torch.tensor([1.0]))
		self.noise_uniform = torch.rand
		# ********************************************************************************************************

		# ********************************************************************************************************
		# FCN FINAL LAYER TO GET VALUES
		self.final_value_layer_1 = nn.Linear(final_input_dim, 64, bias=False)
		self.final_value_layer_2 = nn.Linear(64, final_output_dim, bias=False)
		# ********************************************************************************************************	

		self.place_policies = torch.zeros(self.num_agents,self.num_agents,obs_act_input_dim).to(self.device)
		self.place_actions = torch.ones(self.num_agents,self.num_agents,obs_act_input_dim).to(self.device)
		one_hots = torch.ones(obs_act_input_dim)
		zero_hots = torch.zeros(obs_act_input_dim)

		for j in range(self.num_agents):
			self.place_policies[j][j] = one_hots
			self.place_actions[j][j] = zero_hots

		self.threshold = threshold
		self.obs_act_input_dim = obs_act_input_dim
		# ********************************************************************************************************* 

		self.reset_parameters()


	def reset_parameters(self):
		"""Reinitialize learnable parameters."""
		gain_leaky = nn.init.calculate_gain('leaky_relu')

		nn.init.xavier_uniform_(self.state_embed[0].weight, gain=gain_leaky)
		nn.init.xavier_uniform_(self.state_act_pol_embed[0].weight, gain=gain_leaky)

		nn.init.xavier_uniform_(self.key_layer.weight)
		nn.init.xavier_uniform_(self.query_layer.weight)
		nn.init.xavier_uniform_(self.attention_value_layer.weight)


		nn.init.xavier_uniform_(self.final_value_layer_1.weight, gain=gain_leaky)
		nn.init.xavier_uniform_(self.final_value_layer_2.weight, gain=gain_leaky)



	def forward(self, states, policies, actions):
		# EMBED STATES
		states_embed = self.state_embed(states)
		# KEYS
		key_obs = self.key_layer(states_embed)
		# QUERIES
		query_obs = self.query_layer(states_embed)
		# WEIGHT
		weight = F.softmax(torch.matmul(query_obs,key_obs.transpose(1,2))/math.sqrt(self.d_k_obs_act),dim=-1)
		ret_weight = weight

		obs_actions = torch.cat([states,actions],dim=-1)
		obs_policy = torch.cat([states,policies], dim=-1)
		obs_actions = obs_actions.repeat(1,self.num_agents,1).reshape(obs_actions.shape[0],self.num_agents,self.num_agents,-1)
		obs_policy = obs_policy.repeat(1,self.num_agents,1).reshape(obs_policy.shape[0],self.num_agents,self.num_agents,-1)
		obs_actions_policies = self.place_policies*obs_policy + self.place_actions*obs_actions
		# EMBED STATE ACTION POLICY
		obs_actions_policies_embed = self.state_act_pol_embed(obs_actions_policies)
		attention_values = self.attention_value_layer(obs_actions_policies_embed)
		attention_values = attention_values.repeat(1,self.num_agents,1,1).reshape(attention_values.shape[0],self.num_agents,self.num_agents,self.num_agents,-1)
		
		weight = weight.unsqueeze(-2).repeat(1,1,self.num_agents,1).unsqueeze(-1)
		weighted_attention_values = attention_values*weight
		node_features = torch.sum(weighted_attention_values, dim=-2) + states_embed.repeat(1,self.num_agents,1).reshape(states_embed.shape[0],self.num_agents,self.num_agents,-1)

		Value = F.leaky_relu(self.final_value_layer_1(node_features))
		Value = self.final_value_layer_2(Value)

		return Value, ret_weight




'''
V2 --> LayerNorm after attention values are computed
'''
class StateActionGATCriticWoResConnV2(nn.Module):
	def __init__(self, obs_input_dim, obs_output_dim, obs_act_input_dim, obs_act_output_dim, final_input_dim, final_output_dim, num_agents, num_actions, threshold=0.1):
		super(StateActionGATCriticWoResConnV2, self).__init__()
		
		self.num_agents = num_agents
		self.num_actions = num_actions
		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

		self.state_embed = nn.Sequential(nn.Linear(obs_input_dim, 128), nn.LeakyReLU())
		self.key_layer = nn.Linear(128, obs_output_dim, bias=False)
		self.query_layer = nn.Linear(128, obs_output_dim, bias=False)
		self.state_act_pol_embed = nn.Sequential(nn.Linear(obs_act_input_dim, 128), nn.LeakyReLU())
		self.attention_value_layer = nn.Linear(128, obs_act_output_dim, bias=False)
		self.LayerNorm = nn.LayerNorm(obs_act_output_dim)
		# dimesion of key
		self.d_k_obs_act = obs_output_dim

		# NOISE
		self.noise_normal = torch.distributions.Normal(loc=torch.tensor([0.0]), scale=torch.tensor([1.0]))
		self.noise_uniform = torch.rand
		# ********************************************************************************************************

		# ********************************************************************************************************
		# FCN FINAL LAYER TO GET VALUES
		self.final_value_layer_1 = nn.Linear(final_input_dim, 64, bias=False)
		self.final_value_layer_2 = nn.Linear(64, final_output_dim, bias=False)
		# ********************************************************************************************************	

		self.place_policies = torch.zeros(self.num_agents,self.num_agents,obs_act_input_dim).to(self.device)
		self.place_actions = torch.ones(self.num_agents,self.num_agents,obs_act_input_dim).to(self.device)
		one_hots = torch.ones(obs_act_input_dim)
		zero_hots = torch.zeros(obs_act_input_dim)

		for j in range(self.num_agents):
			self.place_policies[j][j] = one_hots
			self.place_actions[j][j] = zero_hots

		self.threshold = threshold
		self.obs_act_input_dim = obs_act_input_dim
		# ********************************************************************************************************* 

		self.shared_modules = [self.state_embed[0], self.key_layer, self.query_layer, self.state_act_pol_embed[0], self.attention_value_layer, self.final_value_layer_1, self.final_value_layer_2]

		self.reset_parameters()


	def reset_parameters(self):
		"""Reinitialize learnable parameters."""
		gain_leaky = nn.init.calculate_gain('leaky_relu')

		nn.init.xavier_uniform_(self.state_embed[0].weight, gain=gain_leaky)
		nn.init.xavier_uniform_(self.state_act_pol_embed[0].weight, gain=gain_leaky)

		nn.init.xavier_uniform_(self.key_layer.weight)
		nn.init.xavier_uniform_(self.query_layer.weight)
		nn.init.xavier_uniform_(self.attention_value_layer.weight)


		nn.init.xavier_uniform_(self.final_value_layer_1.weight, gain=gain_leaky)
		nn.init.xavier_uniform_(self.final_value_layer_2.weight, gain=gain_leaky)


	def shared_parameters(self):
		"""
		Parameters shared across agents and reward heads
		"""
		return chain(*[m.parameters() for m in self.shared_modules])

	def scale_shared_grads(self):
		"""
		Scale gradients for parameters that are shared since they accumulate
		gradients from the critic loss function multiple times
		"""
		for p in self.shared_parameters():
			p.grad.data.mul_(1. / self.num_agents)



	def forward(self, states, policies, actions):
		# EMBED STATES
		states_embed = self.state_embed(states)
		# KEYS
		key_obs = self.key_layer(states_embed)
		# QUERIES
		query_obs = self.query_layer(states_embed)
		# WEIGHT
		weight = F.softmax(torch.matmul(query_obs,key_obs.transpose(1,2))/math.sqrt(self.d_k_obs_act),dim=-1)
		ret_weight = weight

		obs_actions = torch.cat([states,actions],dim=-1)
		obs_policy = torch.cat([states,policies], dim=-1)
		obs_actions = obs_actions.repeat(1,self.num_agents,1).reshape(obs_actions.shape[0],self.num_agents,self.num_agents,-1)
		obs_policy = obs_policy.repeat(1,self.num_agents,1).reshape(obs_policy.shape[0],self.num_agents,self.num_agents,-1)
		obs_actions_policies = self.place_policies*obs_policy + self.place_actions*obs_actions
		# EMBED STATE ACTION POLICY
		obs_actions_policies_embed = self.state_act_pol_embed(obs_actions_policies)
		attention_values = self.attention_value_layer(obs_actions_policies_embed)
		attention_values = attention_values.repeat(1,self.num_agents,1,1).reshape(attention_values.shape[0],self.num_agents,self.num_agents,self.num_agents,-1)
		
		weight = weight.unsqueeze(-2).repeat(1,1,self.num_agents,1).unsqueeze(-1)
		weighted_attention_values = attention_values*weight
		node_features = self.LayerNorm(torch.sum(weighted_attention_values, dim=-2))

		Value = F.leaky_relu(self.final_value_layer_1(node_features))
		Value = self.final_value_layer_2(Value)

		return Value, ret_weight



class StateActionGATCriticWResConnV2(nn.Module):
	def __init__(self, obs_input_dim, obs_output_dim, obs_act_input_dim, obs_act_output_dim, final_input_dim, final_output_dim, num_agents, num_actions, threshold=0.1):
		super(StateActionGATCriticWResConnV2, self).__init__()
		
		self.num_agents = num_agents
		self.num_actions = num_actions
		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

		self.state_embed = nn.Sequential(nn.Linear(obs_input_dim, 128), nn.LeakyReLU())
		self.key_layer = nn.Linear(128, obs_output_dim, bias=False)
		self.query_layer = nn.Linear(128, obs_output_dim, bias=False)
		self.state_act_pol_embed = nn.Sequential(nn.Linear(obs_act_input_dim, 128), nn.LeakyReLU())
		self.attention_value_layer = nn.Linear(128, obs_act_output_dim, bias=False)
		self.LayerNorm = nn.LayerNorm(obs_act_output_dim)
		# dimesion of key
		self.d_k_obs_act = obs_output_dim

		# NOISE
		self.noise_normal = torch.distributions.Normal(loc=torch.tensor([0.0]), scale=torch.tensor([1.0]))
		self.noise_uniform = torch.rand
		# ********************************************************************************************************

		# ********************************************************************************************************
		# FCN FINAL LAYER TO GET VALUES
		self.final_value_layer_1 = nn.Linear(final_input_dim, 64, bias=False)
		self.final_value_layer_2 = nn.Linear(64, final_output_dim, bias=False)
		# ********************************************************************************************************	

		self.place_policies = torch.zeros(self.num_agents,self.num_agents,obs_act_input_dim).to(self.device)
		self.place_actions = torch.ones(self.num_agents,self.num_agents,obs_act_input_dim).to(self.device)
		one_hots = torch.ones(obs_act_input_dim)
		zero_hots = torch.zeros(obs_act_input_dim)

		for j in range(self.num_agents):
			self.place_policies[j][j] = one_hots
			self.place_actions[j][j] = zero_hots

		self.threshold = threshold
		self.obs_act_input_dim = obs_act_input_dim
		# ********************************************************************************************************* 

		self.reset_parameters()


	def reset_parameters(self):
		"""Reinitialize learnable parameters."""
		gain_leaky = nn.init.calculate_gain('leaky_relu')

		nn.init.xavier_uniform_(self.state_embed[0].weight, gain=gain_leaky)
		nn.init.xavier_uniform_(self.state_act_pol_embed[0].weight, gain=gain_leaky)

		nn.init.xavier_uniform_(self.key_layer.weight)
		nn.init.xavier_uniform_(self.query_layer.weight)
		nn.init.xavier_uniform_(self.attention_value_layer.weight)


		nn.init.xavier_uniform_(self.final_value_layer_1.weight, gain=gain_leaky)
		nn.init.xavier_uniform_(self.final_value_layer_2.weight, gain=gain_leaky)



	def forward(self, states, policies, actions):
		# EMBED STATES
		states_embed = self.state_embed(states)
		# KEYS
		key_obs = self.key_layer(states_embed)
		# QUERIES
		query_obs = self.query_layer(states_embed)
		# WEIGHT
		weight = F.softmax(torch.matmul(query_obs,key_obs.transpose(1,2))/math.sqrt(self.d_k_obs_act),dim=-1)
		ret_weight = weight

		obs_actions = torch.cat([states,actions],dim=-1)
		obs_policy = torch.cat([states,policies], dim=-1)
		obs_actions = obs_actions.repeat(1,self.num_agents,1).reshape(obs_actions.shape[0],self.num_agents,self.num_agents,-1)
		obs_policy = obs_policy.repeat(1,self.num_agents,1).reshape(obs_policy.shape[0],self.num_agents,self.num_agents,-1)
		obs_actions_policies = self.place_policies*obs_policy + self.place_actions*obs_actions
		# EMBED STATE ACTION POLICY
		obs_actions_policies_embed = self.state_act_pol_embed(obs_actions_policies)
		attention_values = self.attention_value_layer(obs_actions_policies_embed)
		attention_values = attention_values.repeat(1,self.num_agents,1,1).reshape(attention_values.shape[0],self.num_agents,self.num_agents,self.num_agents,-1)
		
		weight = weight.unsqueeze(-2).repeat(1,1,self.num_agents,1).unsqueeze(-1)
		weighted_attention_values = attention_values*weight
		node_features = self.LayerNorm(torch.sum(weighted_attention_values, dim=-2) + states_embed.repeat(1,self.num_agents,1).reshape(states_embed.shape[0],self.num_agents,self.num_agents,-1))

		Value = F.leaky_relu(self.final_value_layer_1(node_features))
		Value = self.final_value_layer_2(Value)

		return Value, ret_weight




'''
V3 --> LayerNorm after attention values are computed; add extra MLP before ValueMLP and also add a Residual Connection
'''
class StateActionGATCriticWoResConnV3(nn.Module):
	def __init__(self, obs_input_dim, obs_output_dim, obs_act_input_dim, obs_act_output_dim, final_input_dim, final_output_dim, num_agents, num_actions, threshold=0.1):
		super(StateActionGATCriticWoResConnV3, self).__init__()
		
		self.num_agents = num_agents
		self.num_actions = num_actions
		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

		self.state_embed = nn.Sequential(nn.Linear(obs_input_dim, 128), nn.LeakyReLU())
		self.key_layer = nn.Linear(128, obs_output_dim, bias=False)
		self.query_layer = nn.Linear(128, obs_output_dim, bias=False)
		self.state_act_pol_embed = nn.Sequential(nn.Linear(obs_act_input_dim, 128), nn.LeakyReLU())
		self.attention_value_layer = nn.Linear(128, obs_act_output_dim, bias=False)
		self.LayerNorm = nn.LayerNorm(obs_act_output_dim)
		# dimesion of key
		self.d_k_obs_act = obs_output_dim

		# NOISE
		self.noise_normal = torch.distributions.Normal(loc=torch.tensor([0.0]), scale=torch.tensor([1.0]))
		self.noise_uniform = torch.rand
		# ********************************************************************************************************

		# ********************************************************************************************************
		# FCN INTERMEDIATE LAYERS
		self.intermediate_layer_1 = nn.Linear(obs_act_output_dim, 128, bias=False)
		self.intermediate_layer_2 = nn.Linear(128, obs_act_output_dim, bias=False)
		# ********************************************************************************************************

		# ********************************************************************************************************
		# FCN FINAL LAYER TO GET VALUES
		self.final_value_layer_1 = nn.Linear(final_input_dim, 64, bias=False)
		self.final_value_layer_2 = nn.Linear(64, final_output_dim, bias=False)
		# ********************************************************************************************************	

		self.place_policies = torch.zeros(self.num_agents,self.num_agents,obs_act_input_dim).to(self.device)
		self.place_actions = torch.ones(self.num_agents,self.num_agents,obs_act_input_dim).to(self.device)
		one_hots = torch.ones(obs_act_input_dim)
		zero_hots = torch.zeros(obs_act_input_dim)

		for j in range(self.num_agents):
			self.place_policies[j][j] = one_hots
			self.place_actions[j][j] = zero_hots

		self.threshold = threshold
		self.obs_act_input_dim = obs_act_input_dim
		# ********************************************************************************************************* 

		self.reset_parameters()


	def reset_parameters(self):
		"""Reinitialize learnable parameters."""
		gain_leaky = nn.init.calculate_gain('leaky_relu')

		nn.init.xavier_uniform_(self.state_embed[0].weight, gain=gain_leaky)
		nn.init.xavier_uniform_(self.state_act_pol_embed[0].weight, gain=gain_leaky)

		nn.init.xavier_uniform_(self.key_layer.weight)
		nn.init.xavier_uniform_(self.query_layer.weight)
		nn.init.xavier_uniform_(self.attention_value_layer.weight)

		nn.init.xavier_uniform_(self.intermediate_layer_1.weight, gain=gain_leaky)
		nn.init.xavier_uniform_(self.intermediate_layer_2.weight, gain=gain_leaky)

		nn.init.xavier_uniform_(self.final_value_layer_1.weight, gain=gain_leaky)
		nn.init.xavier_uniform_(self.final_value_layer_2.weight, gain=gain_leaky)



	def forward(self, states, policies, actions):
		# EMBED STATES
		states_embed = self.state_embed(states)
		# KEYS
		key_obs = self.key_layer(states_embed)
		# QUERIES
		query_obs = self.query_layer(states_embed)
		# WEIGHT
		weight = F.softmax(torch.matmul(query_obs,key_obs.transpose(1,2))/math.sqrt(self.d_k_obs_act),dim=-1)
		ret_weight = weight

		obs_actions = torch.cat([states,actions],dim=-1)
		obs_policy = torch.cat([states,policies], dim=-1)
		obs_actions = obs_actions.repeat(1,self.num_agents,1).reshape(obs_actions.shape[0],self.num_agents,self.num_agents,-1)
		obs_policy = obs_policy.repeat(1,self.num_agents,1).reshape(obs_policy.shape[0],self.num_agents,self.num_agents,-1)
		obs_actions_policies = self.place_policies*obs_policy + self.place_actions*obs_actions
		# EMBED STATE ACTION POLICY
		obs_actions_policies_embed = self.state_act_pol_embed(obs_actions_policies)
		attention_values = self.attention_value_layer(obs_actions_policies_embed)
		attention_values = attention_values.repeat(1,self.num_agents,1,1).reshape(attention_values.shape[0],self.num_agents,self.num_agents,self.num_agents,-1)
		
		weight = weight.unsqueeze(-2).repeat(1,1,self.num_agents,1).unsqueeze(-1)
		weighted_attention_values = attention_values*weight
		node_features = self.LayerNorm(torch.sum(weighted_attention_values, dim=-2))

		intermediate_values = F.leaky_relu(self.intermediate_layer_1(node_features))
		intermediate_values = F.leaky_relu(self.intermediate_layer_2(intermediate_values))

		Value = F.leaky_relu(self.final_value_layer_1(intermediate_values))
		Value = self.final_value_layer_2(Value)

		return Value, ret_weight



class StateActionGATCriticWResConnV3(nn.Module):
	def __init__(self, obs_input_dim, obs_output_dim, obs_act_input_dim, obs_act_output_dim, final_input_dim, final_output_dim, num_agents, num_actions, threshold=0.1):
		super(StateActionGATCriticWResConnV3, self).__init__()
		
		self.num_agents = num_agents
		self.num_actions = num_actions
		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

		self.state_embed = nn.Sequential(nn.Linear(obs_input_dim, 128), nn.LeakyReLU())
		self.key_layer = nn.Linear(128, obs_output_dim, bias=False)
		self.query_layer = nn.Linear(128, obs_output_dim, bias=False)
		self.state_act_pol_embed = nn.Sequential(nn.Linear(obs_act_input_dim, 128), nn.LeakyReLU())
		self.attention_value_layer = nn.Linear(128, obs_act_output_dim, bias=False)
		self.LayerNorm = nn.LayerNorm(obs_act_output_dim)
		# dimesion of key
		self.d_k_obs_act = obs_output_dim

		# NOISE
		self.noise_normal = torch.distributions.Normal(loc=torch.tensor([0.0]), scale=torch.tensor([1.0]))
		self.noise_uniform = torch.rand
		# ********************************************************************************************************

		# ********************************************************************************************************
		# FCN INTERMEDIATE LAYERS
		self.intermediate_layer_1 = nn.Linear(obs_act_output_dim, 128, bias=False)
		self.intermediate_layer_2 = nn.Linear(128, obs_act_output_dim, bias=False)
		self.LayerNorm_1 = nn.LayerNorm(obs_act_output_dim)
		# ********************************************************************************************************

		# ********************************************************************************************************
		# FCN FINAL LAYER TO GET VALUES
		self.final_value_layer_1 = nn.Linear(final_input_dim, 64, bias=False)
		self.final_value_layer_2 = nn.Linear(64, final_output_dim, bias=False)
		# ********************************************************************************************************	

		self.place_policies = torch.zeros(self.num_agents,self.num_agents,obs_act_input_dim).to(self.device)
		self.place_actions = torch.ones(self.num_agents,self.num_agents,obs_act_input_dim).to(self.device)
		one_hots = torch.ones(obs_act_input_dim)
		zero_hots = torch.zeros(obs_act_input_dim)

		for j in range(self.num_agents):
			self.place_policies[j][j] = one_hots
			self.place_actions[j][j] = zero_hots

		self.threshold = threshold
		self.obs_act_input_dim = obs_act_input_dim
		# ********************************************************************************************************* 

		self.reset_parameters()


	def reset_parameters(self):
		"""Reinitialize learnable parameters."""
		gain_leaky = nn.init.calculate_gain('leaky_relu')

		nn.init.xavier_uniform_(self.state_embed[0].weight, gain=gain_leaky)
		nn.init.xavier_uniform_(self.state_act_pol_embed[0].weight, gain=gain_leaky)

		nn.init.xavier_uniform_(self.key_layer.weight)
		nn.init.xavier_uniform_(self.query_layer.weight)
		nn.init.xavier_uniform_(self.attention_value_layer.weight)

		nn.init.xavier_uniform_(self.intermediate_layer_1.weight, gain=gain_leaky)
		nn.init.xavier_uniform_(self.intermediate_layer_2.weight, gain=gain_leaky)

		nn.init.xavier_uniform_(self.final_value_layer_1.weight, gain=gain_leaky)
		nn.init.xavier_uniform_(self.final_value_layer_2.weight, gain=gain_leaky)



	def forward(self, states, policies, actions):
		# EMBED STATES
		states_embed = self.state_embed(states)
		# KEYS
		key_obs = self.key_layer(states_embed)
		# QUERIES
		query_obs = self.query_layer(states_embed)
		# WEIGHT
		weight = F.softmax(torch.matmul(query_obs,key_obs.transpose(1,2))/math.sqrt(self.d_k_obs_act),dim=-1)
		ret_weight = weight

		obs_actions = torch.cat([states,actions],dim=-1)
		obs_policy = torch.cat([states,policies], dim=-1)
		obs_actions = obs_actions.repeat(1,self.num_agents,1).reshape(obs_actions.shape[0],self.num_agents,self.num_agents,-1)
		obs_policy = obs_policy.repeat(1,self.num_agents,1).reshape(obs_policy.shape[0],self.num_agents,self.num_agents,-1)
		obs_actions_policies = self.place_policies*obs_policy + self.place_actions*obs_actions
		# EMBED STATE ACTION POLICY
		obs_actions_policies_embed = self.state_act_pol_embed(obs_actions_policies)
		attention_values = self.attention_value_layer(obs_actions_policies_embed)
		attention_values = attention_values.repeat(1,self.num_agents,1,1).reshape(attention_values.shape[0],self.num_agents,self.num_agents,self.num_agents,-1)
		
		weight = weight.unsqueeze(-2).repeat(1,1,self.num_agents,1).unsqueeze(-1)
		weighted_attention_values = torch.sum(attention_values*weight, dim=-2)
		node_features = self.LayerNorm(weighted_attention_values)

		intermediate_values = F.leaky_relu(self.intermediate_layer_1(node_features))
		intermediate_values = F.leaky_relu(self.intermediate_layer_2(intermediate_values))
		intermediate_values = self.LayerNorm_1(intermediate_values+weighted_attention_values)

		Value = F.leaky_relu(self.final_value_layer_1(intermediate_values))
		Value = self.final_value_layer_2(Value)

		return Value, ret_weight




'''
MLP to GNN
'''

class MLPToGNNV1(nn.Module):
	'''
	V1: [s0,s1,s2,s3] fed into MLP
	'''
	def __init__(self,state_dim,num_agents):
		super(MLPToGNNV1,self).__init__()

		self.state_dim = state_dim
		self.num_agents = num_agents		
		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


		self.fc1 = nn.Linear(state_dim*num_agents,128)
		self.fc2 = nn.Linear(128,128)
		self.fc3 = nn.Linear(128,1)

	def forward(self, states, policies=None, actions=None):

		# T x num_agents x state_dim
		T = states.shape[0]
		
		states_aug = [torch.roll(states,i,1) for i in range(self.num_agents)]

		states_aug = torch.cat(states_aug,dim=2)

		x = self.fc1(states_aug)
		x = nn.ReLU()(x)
		x = self.fc2(x)
		x = nn.ReLU()(x)
		V = self.fc3(x)

		V = torch.stack(self.num_agents*[V],2)

		return V, 1/self.num_agents*torch.ones((T,self.num_agents,self.num_agents),device=self.device)



class MLPToGNNV2(nn.Module):
	'''
	V2: s0,...,s3 fed into "value network" mlps, concatenated, fed into final MLP
	'''
	def __init__(self,state_dim,num_agents):
		super(MLPToGNNV2,self).__init__()

		self.state_dim = state_dim
		self.num_agents = num_agents		
		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

		self.value_fc1 = nn.Linear(state_dim,32)

		self.fc1 = nn.Linear(32*num_agents,128)
		self.fc2 = nn.Linear(128,128)
		self.fc3 = nn.Linear(128,1)

	def forward(self, states, policies=None, actions=None):

		states_value = F.leaky_relu(self.value_fc1(states))

		# T x num_agents x state_dim
		T = states_value.shape[0]
		
		states_aug = [torch.roll(states_value,i,1) for i in range(self.num_agents)]

		states_aug = torch.cat(states_aug,dim=2)

		x = self.fc1(states_aug)
		x = nn.ReLU()(x)
		x = self.fc2(x)
		x = nn.ReLU()(x)
		V = self.fc3(x)

		V = torch.stack(self.num_agents*[V],2)

		return V, 1/self.num_agents*torch.ones((T,self.num_agents,self.num_agents),device=self.device)



class MLPToGNNV3(nn.Module):
	'''
	V3: s0,...,s3 fed into value networks to get values, attention weights computed, weighted values concatenated, fed into final mlp
	'''
	def __init__(self,state_dim,num_agents):
		super(MLPToGNNV3,self).__init__()

		self.state_dim = state_dim
		self.num_agents = num_agents		
		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

		self.key = nn.Linear(state_dim,64)
		self.query = nn.Linear(state_dim,64)
		self.attention_value = nn.Linear(state_dim,32)

		self.d_k_obs = 64

		self.fc1 = nn.Linear(32*num_agents,128)
		self.fc2 = nn.Linear(128,128)
		self.fc3 = nn.Linear(128,1)

	def forward(self, states, policies=None, actions=None):
		keys = self.key(states)
		queries = self.query(states)
		weights = F.softmax(torch.matmul(queries,keys.transpose(1,2))/math.sqrt(self.d_k_obs),dim=-1).unsqueeze(-1)
		attention_values = F.leaky_relu(self.attention_value(states)).unsqueeze(-2).repeat(1,1,self.num_agents,1)

		weighted_attention_values = weights*attention_values
		# T x N x D
		node_features = weighted_attention_values.reshape(weighted_attention_values.shape[0],self.num_agents,-1)
		
		x = self.fc1(node_features)
		x = nn.ReLU()(x)
		x = self.fc2(x)
		x = nn.ReLU()(x)
		V = self.fc3(x)

		V = torch.stack(self.num_agents*[V],2)

		return V, weights



class MLPToGNNV4(nn.Module):
	'''
	V4: same as V3, but action/policies are used to get values
	'''
	def __init__(self,state_dim,action_dim,num_agents):
		super(MLPToGNNV4,self).__init__()

		self.state_dim = state_dim
		self.num_agents = num_agents		
		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

		self.key = nn.Linear(state_dim,64)
		self.query = nn.Linear(state_dim,64)
		self.attention_value = nn.Linear((state_dim+action_dim),32)

		self.d_k_obs_act = 64

		self.fc1 = nn.Linear(32*num_agents,128)
		self.fc2 = nn.Linear(128,128)
		self.fc3 = nn.Linear(128,1)


		self.place_policies = torch.zeros(self.num_agents,self.num_agents,state_dim+action_dim).to(self.device)
		self.place_actions = torch.ones(self.num_agents,self.num_agents,state_dim+action_dim).to(self.device)
		one_hots = torch.ones(state_dim+action_dim)
		zero_hots = torch.zeros(state_dim+action_dim)

		for j in range(self.num_agents):
			self.place_policies[j][j] = one_hots
			self.place_actions[j][j] = zero_hots

	def forward(self, states, policies=None, actions=None):
		keys = self.key(states)
		queries = self.query(states)
		weights = F.softmax(torch.matmul(queries,keys.transpose(1,2))/math.sqrt(self.d_k_obs_act),dim=-1)

		obs_actions = torch.cat([states,actions],dim=-1)
		obs_policy = torch.cat([states,policies], dim=-1)
		obs_actions = obs_actions.repeat(1,self.num_agents,1).reshape(obs_actions.shape[0],self.num_agents,self.num_agents,-1)
		obs_policy = obs_policy.repeat(1,self.num_agents,1).reshape(obs_policy.shape[0],self.num_agents,self.num_agents,-1)
		obs_actions_policies = self.place_policies*obs_policy + self.place_actions*obs_actions
		attention_values = F.leaky_relu(self.attention_value(obs_actions_policies))
		attention_values = attention_values.repeat(1,self.num_agents,1,1).reshape(attention_values.shape[0],self.num_agents,self.num_agents,self.num_agents,-1)

		weighted_attention_values = weights.unsqueeze(-2).repeat(1,1,self.num_agents,1).unsqueeze(-1)*attention_values
		# T x N x N x D
		node_features = weighted_attention_values.reshape(weighted_attention_values.shape[0],self.num_agents, self.num_agents, -1)
		
		x = self.fc1(node_features)
		x = nn.ReLU()(x)
		x = self.fc2(x)
		x = nn.ReLU()(x)
		V = self.fc3(x)

		return V, weights



class MLPToGNNV5(nn.Module):
	'''
	V5: same as V4, but weighted values are summed instead of concatenated
	'''
	def __init__(self, obs_input_dim, obs_output_dim, obs_act_input_dim, obs_act_output_dim, final_input_dim, final_output_dim, num_agents, num_actions, threshold=0.1):
		super(MLPToGNNV5, self).__init__()
		
		self.num_agents = num_agents
		self.num_actions = num_actions
		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

		self.state_embed = nn.Sequential(nn.Linear(obs_input_dim, 128), nn.LeakyReLU())
		self.key_layer = nn.Linear(128, obs_output_dim, bias=False)
		self.query_layer = nn.Linear(128, obs_output_dim, bias=False)
		self.state_act_pol_embed = nn.Sequential(nn.Linear(obs_act_input_dim, 128), nn.LeakyReLU())
		self.attention_value_layer = nn.Linear(128, obs_act_output_dim, bias=False)
		self.LayerNorm = nn.LayerNorm(obs_act_output_dim)
		# dimesion of key
		self.d_k_obs_act = obs_output_dim  

		# NOISE
		self.noise_normal = torch.distributions.Normal(loc=torch.tensor([0.0]), scale=torch.tensor([1.0]))
		self.noise_uniform = torch.rand
		# ********************************************************************************************************

		# ********************************************************************************************************
		# FCN FINAL LAYER TO GET VALUES
		self.final_value_layer_1 = nn.Linear(final_input_dim, 64, bias=False)
		self.final_value_layer_2 = nn.Linear(64, final_output_dim, bias=False)
		# ********************************************************************************************************	

		self.place_policies = torch.zeros(self.num_agents,self.num_agents,obs_act_input_dim).to(self.device)
		self.place_actions = torch.ones(self.num_agents,self.num_agents,obs_act_input_dim).to(self.device)
		one_hots = torch.ones(obs_act_input_dim)
		zero_hots = torch.zeros(obs_act_input_dim)

		for j in range(self.num_agents):
			self.place_policies[j][j] = one_hots
			self.place_actions[j][j] = zero_hots

		self.threshold = threshold
		self.obs_act_input_dim = obs_act_input_dim
		# ********************************************************************************************************* 

		self.shared_modules = [self.state_embed[0], self.key_layer, self.query_layer, self.state_act_pol_embed[0], self.attention_value_layer, self.final_value_layer_1, self.final_value_layer_2]

		self.reset_parameters()


	def reset_parameters(self):
		"""Reinitialize learnable parameters."""
		gain_leaky = nn.init.calculate_gain('leaky_relu')

		nn.init.xavier_uniform_(self.state_embed[0].weight, gain=gain_leaky)
		nn.init.xavier_uniform_(self.state_act_pol_embed[0].weight, gain=gain_leaky)

		nn.init.xavier_uniform_(self.key_layer.weight)
		nn.init.xavier_uniform_(self.query_layer.weight)
		nn.init.xavier_uniform_(self.attention_value_layer.weight)


		nn.init.xavier_uniform_(self.final_value_layer_1.weight, gain=gain_leaky)
		nn.init.xavier_uniform_(self.final_value_layer_2.weight, gain=gain_leaky)


	def shared_parameters(self):
		"""
		Parameters shared across agents and reward heads
		"""
		return chain(*[m.parameters() for m in self.shared_modules])

	def scale_shared_grads(self):
		"""
		Scale gradients for parameters that are shared since they accumulate
		gradients from the critic loss function multiple times
		"""
		for p in self.shared_parameters():
			p.grad.data.mul_(1. / self.num_agents)



	def forward(self, states, policies, actions):
		# EMBED STATES
		states_embed = self.state_embed(states)
		# KEYS
		key_obs = self.key_layer(states_embed)
		# QUERIES
		query_obs = self.query_layer(states_embed)
		# WEIGHT
		weight = F.softmax(torch.matmul(query_obs,key_obs.transpose(1,2))/math.sqrt(self.d_k_obs_act),dim=-1)
		ret_weight = weight

		obs_actions = torch.cat([states,actions],dim=-1)
		obs_policy = torch.cat([states,policies], dim=-1)
		obs_actions = obs_actions.repeat(1,self.num_agents,1).reshape(obs_actions.shape[0],self.num_agents,self.num_agents,-1)
		obs_policy = obs_policy.repeat(1,self.num_agents,1).reshape(obs_policy.shape[0],self.num_agents,self.num_agents,-1)
		obs_actions_policies = self.place_policies*obs_policy + self.place_actions*obs_actions
		# EMBED STATE ACTION POLICY
		obs_actions_policies_embed = self.state_act_pol_embed(obs_actions_policies)
		attention_values = self.attention_value_layer(obs_actions_policies_embed)
		attention_values = attention_values.repeat(1,self.num_agents,1,1).reshape(attention_values.shape[0],self.num_agents,self.num_agents,self.num_agents,-1)
		
		weight = weight.unsqueeze(-2).repeat(1,1,self.num_agents,1).unsqueeze(-1)
		weighted_attention_values = attention_values*weight
		node_features = self.LayerNorm(torch.sum(weighted_attention_values, dim=-2))

		Value = F.leaky_relu(self.final_value_layer_1(node_features))
		Value = self.final_value_layer_2(Value)

		return Value, ret_weight



class MLPToGNNV6(nn.Module):
	'''
	V6: same as V5, but without LayerNorm
	'''
	def __init__(self, obs_input_dim, obs_output_dim, obs_act_input_dim, obs_act_output_dim, final_input_dim, final_output_dim, num_agents, num_actions, threshold=0.1):
		super(MLPToGNNV6, self).__init__()
		
		self.num_agents = num_agents
		self.num_actions = num_actions
		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

		self.state_embed = nn.Sequential(nn.Linear(obs_input_dim, 128), nn.LeakyReLU())
		self.key_layer = nn.Linear(128, obs_output_dim, bias=False)
		self.query_layer = nn.Linear(128, obs_output_dim, bias=False)
		self.state_act_pol_embed = nn.Sequential(nn.Linear(obs_act_input_dim, 128), nn.LeakyReLU())
		self.attention_value_layer = nn.Linear(128, obs_act_output_dim, bias=False)
		# dimesion of key
		self.d_k_obs_act = obs_output_dim  

		# NOISE
		self.noise_normal = torch.distributions.Normal(loc=torch.tensor([0.0]), scale=torch.tensor([1.0]))
		self.noise_uniform = torch.rand
		# ********************************************************************************************************

		# ********************************************************************************************************
		# FCN FINAL LAYER TO GET VALUES
		self.final_value_layer_1 = nn.Linear(final_input_dim, 64, bias=False)
		self.final_value_layer_2 = nn.Linear(64, final_output_dim, bias=False)
		# ********************************************************************************************************	

		self.place_policies = torch.zeros(self.num_agents,self.num_agents,obs_act_input_dim).to(self.device)
		self.place_actions = torch.ones(self.num_agents,self.num_agents,obs_act_input_dim).to(self.device)
		one_hots = torch.ones(obs_act_input_dim)
		zero_hots = torch.zeros(obs_act_input_dim)

		for j in range(self.num_agents):
			self.place_policies[j][j] = one_hots
			self.place_actions[j][j] = zero_hots

		self.threshold = threshold
		self.obs_act_input_dim = obs_act_input_dim
		# ********************************************************************************************************* 

		self.shared_modules = [self.state_embed[0], self.key_layer, self.query_layer, self.state_act_pol_embed[0], self.attention_value_layer, self.final_value_layer_1, self.final_value_layer_2]

		self.reset_parameters()


	def reset_parameters(self):
		"""Reinitialize learnable parameters."""
		gain_leaky = nn.init.calculate_gain('leaky_relu')

		nn.init.xavier_uniform_(self.state_embed[0].weight, gain=gain_leaky)
		nn.init.xavier_uniform_(self.state_act_pol_embed[0].weight, gain=gain_leaky)

		nn.init.xavier_uniform_(self.key_layer.weight)
		nn.init.xavier_uniform_(self.query_layer.weight)
		nn.init.xavier_uniform_(self.attention_value_layer.weight)


		nn.init.xavier_uniform_(self.final_value_layer_1.weight, gain=gain_leaky)
		nn.init.xavier_uniform_(self.final_value_layer_2.weight, gain=gain_leaky)


	def shared_parameters(self):
		"""
		Parameters shared across agents and reward heads
		"""
		return chain(*[m.parameters() for m in self.shared_modules])

	def scale_shared_grads(self):
		"""
		Scale gradients for parameters that are shared since they accumulate
		gradients from the critic loss function multiple times
		"""
		for p in self.shared_parameters():
			p.grad.data.mul_(1. / self.num_agents)



	def forward(self, states, policies, actions):
		# EMBED STATES
		states_embed = self.state_embed(states)
		# KEYS
		key_obs = self.key_layer(states_embed)
		# QUERIES
		query_obs = self.query_layer(states_embed)
		# WEIGHT
		weight = F.softmax(torch.matmul(query_obs,key_obs.transpose(1,2))/math.sqrt(self.d_k_obs_act),dim=-1)
		ret_weight = weight

		obs_actions = torch.cat([states,actions],dim=-1)
		obs_policy = torch.cat([states,policies], dim=-1)
		obs_actions = obs_actions.repeat(1,self.num_agents,1).reshape(obs_actions.shape[0],self.num_agents,self.num_agents,-1)
		obs_policy = obs_policy.repeat(1,self.num_agents,1).reshape(obs_policy.shape[0],self.num_agents,self.num_agents,-1)
		obs_actions_policies = self.place_policies*obs_policy + self.place_actions*obs_actions
		# EMBED STATE ACTION POLICY
		obs_actions_policies_embed = self.state_act_pol_embed(obs_actions_policies)
		attention_values = self.attention_value_layer(obs_actions_policies_embed)
		attention_values = attention_values.repeat(1,self.num_agents,1,1).reshape(attention_values.shape[0],self.num_agents,self.num_agents,self.num_agents,-1)
		
		weight = weight.unsqueeze(-2).repeat(1,1,self.num_agents,1).unsqueeze(-1)
		weighted_attention_values = attention_values*weight
		node_features = torch.sum(weighted_attention_values, dim=-2)

		Value = F.leaky_relu(self.final_value_layer_1(node_features))
		Value = self.final_value_layer_2(Value)

		return Value, ret_weight