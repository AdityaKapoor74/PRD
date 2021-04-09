import numpy as np
import torch 
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
from torch.autograd import Variable
from torch.distributions import Categorical
from a2c_v2_decoupled import PolicyNetwork, CriticNetwork, WeightNetwork
import torch.nn.functional as F
from torch.utils.data import DataLoader

class A2CAgent:

	def __init__(
		self, 
		env, 
		value_lr=2e-4,
		weight_lr=2e-4, 
		policy_lr=2e-4, 
		entropy_pen=0.008, 
		gamma=0.99,
		lambda_pen = 0.0,
		trace_decay = 0.98,
		gif = False
		):

		self.env = env
		self.value_lr = value_lr
		self.weight_lr = weight_lr
		self.policy_lr = policy_lr
		self.gamma = gamma
		self.entropy_pen = entropy_pen
		self.lambda_pen = lambda_pen
		self.trace_decay = trace_decay
		self.tau = 0.999

		# self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		self.device = "cpu"
		
		self.num_agents = self.env.n
		self.num_actions = self.env.action_space[0].n
		self.gif = gif

		self.weight_input_dim = 2*3+2 # (pose,vel,goal pose, paired agent goal pose) --> observations
		self.weight_output_dim = 16
		self.obs_z_input_dim = 2*3+2 + self.num_actions
		self.obs_z_output_dim = 32
		self.final_input_dim = self.obs_z_output_dim + self.weight_input_dim
		self.final_output_dim = 1
		self.weight_network = WeightNetwork(self.weight_input_dim, self.weight_output_dim, self.obs_z_input_dim, self.obs_z_output_dim, self.final_input_dim, self.final_output_dim, self.num_agents, self.num_actions).to(self.device)
		self.critic_network = CriticNetwork(self.obs_z_input_dim, self.obs_z_output_dim, self.final_input_dim, self.final_output_dim, self.num_agents, self.num_actions).to(self.device)

		self.policy_input_dim = 2*(3+2*(self.num_agents-1)) #2 for pose, 2 for vel and 2 for goal of current agent and rest (2 each) for relative position and relative velocity of other agents
		self.policy_output_dim = self.env.action_space[0].n
		policy_network_size = (self.policy_input_dim,512,256,self.policy_output_dim)
		self.policy_network = PolicyNetwork(policy_network_size).to(self.device)



		# Loading models
		# model_path_value = "../../models/Experiment2/critic_networks/25-01-2021_VN_GAT1_PREPROC_GAT1_FC1_lr0.0002_PN_FC2_lr0.0002_GradNorm0.5_Entropy0.008_lambda0.1_epsiode46000.pt"
		# model_path_policy = "../../models/Experiment2/actor_networks/25-01-2021_PN_FC2_lr0.0002_VN_GAT1_PREPROC_GAT1_FC1_lr0.0002_GradNorm0.5_Entropy0.008_lambda0.1_epsiode46000.pt"
		# For CPU
		# self.critic_network.load_state_dict(torch.load(model_path_value,map_location=torch.device('cpu')))
		# self.policy_network.load_state_dict(torch.load(model_path_policy,map_location=torch.device('cpu')))
		# # For GPU
		# self.critic_network.load_state_dict(torch.load(model_path_value))
		# self.policy_network.load_state_dict(torch.load(model_path_policy))


		self.critic_optimizer = optim.Adam(self.critic_network.parameters(),lr=self.value_lr)
		self.policy_optimizer = optim.Adam(self.policy_network.parameters(),lr=self.policy_lr)
		self.weight_optimizer = optim.Adam(self.weight_network.parameters(),lr=self.weight_lr)


	def get_action(self,state):
		state = torch.FloatTensor(state).to(self.device)
		dists = self.policy_network.forward(state)
		probs = Categorical(dists)
		index = probs.sample().cpu().detach().item()

		return index



	def calculate_advantages(self,returns, values, rewards, dones, GAE = False, normalize = False):
		
		advantages = None

		if GAE:
			advantages = []
			next_value = 0
			advantage = 0
			rewards = rewards.unsqueeze(-1)
			dones = dones.unsqueeze(-1)
			masks = 1 - dones
			for t in reversed(range(0, len(rewards))):
				td_error = rewards[t] + (self.gamma * next_value * masks[t]) - values.data[t]
				next_value = values.data[t]
				
				advantage = td_error + (self.gamma * self.trace_decay * advantage * masks[t])
				advantages.insert(0, advantage)

			advantages = torch.stack(advantages)	
		else:
			advantages = returns - values
		
		if normalize:
			
			advantages = (advantages - advantages.mean()) / advantages.std()
		
		return advantages


	def calculate_returns(self,rewards, discount_factor, normalize = False):
	
		returns = []
		R = 0
		
		for r in reversed(rewards):
			R = r + R * discount_factor
			returns.insert(0, R)
		
		returns_tensor = torch.stack(returns).to(self.device)
		
		if normalize:
			
			returns_tensor = (returns_tensor - returns_tensor.mean()) / returns_tensor.std()
			
		return returns_tensor
		
		


	def update(self,states_critic,next_states_critic,one_hot_actions,one_hot_next_actions,actions,states_actor,next_states_actor,rewards,dones):

		'''
		Getting the probability mass function over the action space for each agent
		'''
		# probs = self.policy_network.forward(actor_graphs).reshape(-1,self.num_agents,self.num_actions)
		probs = self.policy_network.forward(states_actor)
		# next_probs = self.policy_network.forward(next_states_actor)

		'''
		Calculate V values
		'''
		Immediate_rewards, weights_z, weights_net_weights_proc = self.weight_network.forward(states_critic, probs.detach(), one_hot_actions)
		Values, critic_net_weights_proc = self.critic_network.forward(states_critic, probs.detach(), one_hot_actions, weights_z.detach())
		
		Immediate_rewards = Immediate_rewards.reshape(-1,self.num_agents,self.num_agents)
		Values = Values.reshape(-1,self.num_agents,self.num_agents)

	# # ***********************************************************************************
	# 	#update critic (value_net)
	# we need a TxNxN vector so inflate the discounted rewards by N --> cloning the discounted rewards for an agent N times
		discounted_rewards = self.calculate_returns(rewards,self.gamma).unsqueeze(-2).repeat(1,self.num_agents,1)
		discounted_rewards = torch.transpose(discounted_rewards,-1,-2)
		value_loss = F.smooth_l1_loss(Values,discounted_rewards)
		weight_loss = F.smooth_l1_loss(Immediate_rewards,torch.transpose(rewards.unsqueeze(-2).repeat(1,self.num_agents,1),-1,-2)) + self.lambda_pen*torch.sum(weights_z)
		# # ***********************************************************************************
	# 	#update actor (policy net)
	# # ***********************************************************************************
		entropy = -torch.mean(torch.sum(probs * torch.log(torch.clamp(probs, 1e-10,1.0)), dim=2))

		# summing across each agent j to get the advantage
		# so we sum across the second last dimension which does A[t,j] = sum(V[t,i,j] - discounted_rewards[t,i])
		advantage = torch.sum(self.calculate_advantages(discounted_rewards, Values, rewards, dones, True, False),dim=-2)
		probs = Categorical(probs)
		policy_loss = -probs.log_prob(actions) * advantage.detach()
		policy_loss = policy_loss.mean() - self.entropy_pen*entropy
	# # ***********************************************************************************
		
	# **********************************

		self.weight_optimizer.zero_grad()
		weight_loss.backward(retain_graph=True)
		grad_norm_weight = torch.nn.utils.clip_grad_norm_(self.critic_network.parameters(),0.5)
		self.weight_optimizer.step()


		self.critic_optimizer.zero_grad()
		value_loss.backward(retain_graph=True)
		grad_norm_value = torch.nn.utils.clip_grad_norm_(self.critic_network.parameters(),0.5)
		self.critic_optimizer.step()


		self.policy_optimizer.zero_grad()
		policy_loss.backward(retain_graph=False)
		grad_norm_policy = torch.nn.utils.clip_grad_norm_(self.policy_network.parameters(),0.5)
		self.policy_optimizer.step()


		return value_loss,weight_loss,policy_loss,entropy,grad_norm_value,grad_norm_weight,grad_norm_policy,weights_z,weights_net_weights_proc,critic_net_weights_proc