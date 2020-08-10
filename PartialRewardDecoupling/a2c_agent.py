import numpy as np
import torch 
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
from torch.autograd import Variable
from torch.distributions import Categorical
from a2c import *
import torch.nn.functional as F
import gc

class A2CAgent:

	def __init__(self,env,value_lr=2e-4, policy_lr=2e-4, gamma=0.99):
		self.env = env
		self.value_lr = value_lr
		self.policy_lr = policy_lr
		self.gamma = gamma

		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		
		self.num_agents = self.env.n

		'''
		VALUE NETWORK:
		input: observation+necessary actions (other agents)
		ouput: quality of the action taken by the agent

		POLICY NETWORK:
		input: observation
		output: probability distribution over the action space

	'''

		self.value_input_dim = self.env.observation_space[0].shape[0]
		self.value_output_dim = 1
		self.weight_output_dim = 2
		self.num_actions = self.env.action_space[0].n
		self.policy_input_dim = self.env.observation_space[0].shape[0]
		self.policy_output_dim = self.env.action_space[0].n

		
		
		self.value_network = ValueNetwork(self.value_input_dim,self.num_agents,self.num_actions,self.weight_output_dim,self.value_output_dim).to(self.device)
		self.policy_network = PolicyNetwork(self.policy_input_dim,self.policy_output_dim).to(self.device)

		'''
		Loading models
		'''

# 		model_path_value = "./models/4_agents/value_net_lr_2e-4_policy_lr_2e-4_with_grad_norm_0.5_entropy_pen_0.008_xavier_uniform_init_clamp_logs.pt"
# 		model_path_policy = "./models/4_agents/policy_net_lr_2e-4_value_lr_2e-4_with_grad_norm_0.5_entropy_pen_0.008_xavier_uniform_init_clamp_logs.pt"
# 		self.value_network.load_state_dict(torch.load(model_path_value,map_location=torch.device('cpu')))
# 		self.policy_network.load_state_dict(torch.load(model_path_policy,map_location=torch.device('cpu')))
#     For high bay
#     self.value_network.load_state_dict(torch.load(model_path_value))
# 		self.policy_network.load_state_dict(torch.load(model_path_policy))


		self.value_optimizer = optim.Adam(self.value_network.parameters(),lr=self.value_lr)
		self.policy_optimizer = optim.Adam(self.policy_network.parameters(),lr=self.policy_lr)
	

	def get_action(self,state):
		state = torch.FloatTensor(state).to(self.device)
		dists = self.policy_network.forward(state)
		probs = Categorical(dists)
		index = probs.sample().cpu().detach().item()

		return index

	def get_one_hot_encoding(self,actions):
		one_hot = torch.zeros([actions.shape[0], self.num_agents, self.env.action_space[0].n], dtype=torch.int32)
		for i in range(one_hot.shape[0]):
			for j in range(self.num_agents):
				one_hot[i][j][int(actions[i][j].item())] = 1

		return one_hot


	def update(self,states,next_states,actions,rewards):

		weights = self.value_network.forward(states,None)
		weights = weights.permute(0,2,1)
		
		weight_prob = weights[0][0]
		weight_action = weights[0][1]

		for i in range(1,weights.shape[0]):
			weight_prob = torch.cat([weight_prob,weights[i][0]])
			weight_action = torch.cat([weight_action,weights[i][1]])

		weight_prob = weight_prob.reshape(-1,self.num_agents)
		weight_action = weight_action.reshape(-1,self.num_agents)

		probs = self.policy_network.forward(states)

		one_hot_actions = self.get_one_hot_encoding(actions)

		weight_prob = weight_prob.unsqueeze(-1)
		weight_prob = weight_prob.expand(weight_prob.shape[0],weight_prob.shape[1],5)
		weight_action = weight_action.unsqueeze(-1)
		weight_action = weight_action.expand(weight_action.shape[0],weight_action.shape[1],5)
		z = probs.cpu()*weight_prob.cpu()+one_hot_actions*weight_action.cpu()
		z = z.detach().numpy()



		states_value = []
		for k in range(states.shape[0]):
			for j in range(states.shape[1]): # states.shape[1]==self.num_agents
				z_copy = z[k].copy()
				z_copy = np.delete(z_copy,(j), axis=0)
				tmp = np.copy(states.cpu()[k][j])
				for i in range(self.num_agents-1):
					if i == self.num_agents-2:
						tmp = np.append(tmp,z_copy[i])
					else:
						tmp = np.insert(tmp,-(self.num_agents-2-i),z_copy[i])
				states_value.append(tmp)
		
		states_value = torch.FloatTensor([state_val for state_val in states_value]).to(self.device).reshape(states.shape[0],states.shape[1],-1)


		
	# ***********************************************************************************
		#update critic (value_net)
		curr_Q = self.value_network.forward(None,states_value)

		discounted_rewards = np.asarray([[torch.sum(torch.FloatTensor([self.gamma**i for i in range(rewards[k][j:].size(0))])* rewards[k][j:]) for j in range(rewards.size(0))] for k in range(self.num_agents)])
		discounted_rewards = np.transpose(discounted_rewards)
	# ***********************************************************************************

	# ***********************************************************************************
		value_targets = rewards + torch.FloatTensor(discounted_rewards).to(self.device)
		value_targets = value_targets.unsqueeze(dim=-1)
		value_loss = F.smooth_l1_loss(curr_Q,value_targets)
	# ***********************************************************************************

		#update actor (policy net)
	# ***********************************************************************************

		entropy = -torch.mean(torch.sum(probs * torch.log(torch.clamp(probs, 1e-10,1.0)), dim=2))

		advantage = value_targets - curr_Q
		probs = Categorical(probs)
		policy_loss = -probs.log_prob(actions).unsqueeze(dim=-1) * advantage.detach()
		policy_loss = policy_loss.mean() - 0.008*entropy
	# ***********************************************************************************
		
	# ***********************************************************************************
		self.value_optimizer.zero_grad()
		value_loss.backward(retain_graph=False)
		grad_norm_value = torch.nn.utils.clip_grad_norm_(self.value_network.parameters(),0.5)
		self.value_optimizer.step()
		

		self.policy_optimizer.zero_grad()
		policy_loss.backward(retain_graph=False)
		grad_norm_policy = torch.nn.utils.clip_grad_norm_(self.policy_network.parameters(),0.5)
		self.policy_optimizer.step()
	# ***********************************************************************************
		return value_loss,policy_loss,entropy,grad_norm_value,grad_norm_policy

