import numpy as np
import torch 
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
from torch.autograd import Variable
from torch.distributions import Categorical
from dual_network_attention import PolicyNetwork, ValueNetwork, AdvantageNetwork, ScalarDotProductCriticNetwork
import torch.nn.functional as F
import dgl

class A2CAgent:

	def __init__(
		self, 
		env, 
		value_i_lr=1e-2, #1e-2 for environment 1
		value_i_j_lr = 1e-2,
		advantage_lr=1e-2,
		policy_lr=2e-4, 
		entropy_pen=0.008, 
		gamma=0.99,
		trace_decay = 0.98,
		tau = 0.999,
		select_above_threshold = 0.1,
		softmax_cut_threshold = 0.1,
		top_k = 2,
		gif = False
		):

		self.env = env
		self.value_i_lr = value_i_lr
		self.value_i_j_lr = value_i_j_lr
		self.advantage_lr = advantage_lr
		self.policy_lr = policy_lr
		self.gamma = gamma
		self.entropy_pen = entropy_pen
		self.trace_decay = trace_decay
		self.tau = tau
		self.top_k = top_k
		# Used for masking advantages above a threshold
		self.select_above_threshold = select_above_threshold
		# cut the tail of softmax --> Used in softmax with normalization
		self.softmax_cut_threshold = softmax_cut_threshold

		# self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		self.device = "cpu"
		
		self.num_agents = self.env.n
		self.num_actions = self.env.action_space[0].n
		self.gif = gif

		# ENVIRONMENT 1
		self.obs_input_dim = 2*4
		self.obs_out_dim = 16
		self.obs_act_input_dim = self.obs_input_dim + self.num_actions # (pose,vel,goal pose, paired agent goal pose) --> observations 
		self.obs_act_output_dim = 16
		self.final_input_dim = self.obs_input_dim + self.obs_act_output_dim #self.obs_z_output_dim + self.weight_input_dim
		self.final_output_dim = 1
		
		# SCALAR DOT PRODUCT
		# V_i_j
		self.critic_network = ScalarDotProductCriticNetwork(self.obs_act_input_dim, self.obs_act_output_dim, self.final_input_dim, self.final_output_dim, self.num_agents, self.num_actions, self.softmax_cut_threshold).to(self.device)
		# V_i
		self.value_network = ValueNetwork(self.obs_input_dim, self.obs_out_dim, self.obs_out_dim, self.final_output_dim, self.num_agents, self.num_actions, self.softmax_cut_threshold).to(self.device)
		# Adv_i_j
		self.advantage_network = AdvantageNetwork(self.obs_act_input_dim, self.obs_act_output_dim, self.obs_act_output_dim, self.final_output_dim, self.num_agents, self.num_actions, self.softmax_cut_threshold).to(self.device)


		self.policy_input_dim = 2*(3+2*(self.num_agents-1)) #2 for pose, 2 for vel and 2 for goal of current agent and rest (2 each) for relative position and relative velocity of other agents
		self.policy_output_dim = self.env.action_space[0].n
		policy_network_size = (self.policy_input_dim,512,256,self.policy_output_dim)
		self.policy_network = PolicyNetwork(policy_network_size).to(self.device)

		self.self_loop = torch.ones(self.num_agents,self.num_agents)
		for i in range(self.self_loop.shape[0]):
			self.self_loop[i][i] = 0


		# Loading models
		# model_path_value = "../../../models/Experiment37/critic_networks/11-04-2021VN_SAT_SAT_FCN_lr0.0002_PN_FCN_lr0.0002_GradNorm0.5_Entropy0.008_trace_decay0.98_lambda0.0tanh_epsiode6000.pt"
		# model_path_policy = "../../../models/Experiment29/actor_networks/09-04-2021_PN_FCN_lr0.0002VN_SAT_SAT_FCN_lr0.0002_GradNorm0.5_Entropy0.008_trace_decay0.98_lambda1e-05tanh_epsiode24000.pt"
		# For CPU
		# self.critic_network.load_state_dict(torch.load(model_path_value,map_location=torch.device('cpu')))
		# self.policy_network.load_state_dict(torch.load(model_path_policy,map_location=torch.device('cpu')))
		# # For GPU
		# self.critic_network.load_state_dict(torch.load(model_path_value))
		# self.policy_network.load_state_dict(torch.load(model_path_policy))


		self.critic_optimizer = optim.Adam(self.critic_network.parameters(),lr=self.value_i_j_lr)
		self.value_optimizer = optim.Adam(self.value_network.parameters(),lr=self.value_i_lr)
		self.adv_optimizer = optim.Adam(self.advantage_network.parameters(),lr=self.advantage_lr)
		self.policy_optimizer = optim.Adam(self.policy_network.parameters(),lr=self.policy_lr)


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
		V_values_i_j, weights_v_i_j = self.critic_network.forward(states_critic, probs.detach(), one_hot_actions)
		# V_values_next, _ = self.critic_network.forward(next_states_critic, next_probs.detach(), one_hot_next_actions)
		V_values_i_j = V_values_i_j.reshape(-1,self.num_agents,self.num_agents)
		# V_values_next = V_values_next.reshape(-1,self.num_agents,self.num_agents)

		V_values_i, weights_v_i = self.value_network.forward(states_critic)
		V_values_i = V_values_i.reshape(-1,self.num_agents)
		Advantage_i, weights_adv_i = self.advantage_network.forward(states_critic, one_hot_actions)
		Advantage_i = Advantage_i.reshape(-1,self.num_agents)

	# # ***********************************************************************************
	# 	#update critic (value_net)
	# we need a TxNxN vector so inflate the discounted rewards by N --> cloning the discounted rewards for an agent N times
		discounted_rewards = self.calculate_returns(rewards,self.gamma)
		
		value_i_loss = F.smooth_l1_loss(V_values_i, discounted_rewards)
		# print("value_i_loss", value_i_loss)

		advantage_target = discounted_rewards - V_values_i.detach()
		# print("Values_i")
		# print(V_values_i)
		# print("Discounted Rewards")
		# print(discounted_rewards)
		# print("Advantage")
		# print(advantage_target)
		advantage_loss = F.smooth_l1_loss(advantage_target, Advantage_i)
		# print("advantage_loss", advantage_loss)

		discounted_rewards = discounted_rewards.unsqueeze(-2).repeat(1,self.num_agents,1)
		discounted_rewards = torch.transpose(discounted_rewards,-1,-2)
		value_i_j_loss = F.smooth_l1_loss(V_values_i_j,discounted_rewards)
		# print("value_i_j_loss", value_i_j_loss)
		
		# # ***********************************************************************************
	# 	#update actor (policy net)
	# # ***********************************************************************************
		entropy = -torch.mean(torch.sum(probs * torch.log(torch.clamp(probs, 1e-10,1.0)), dim=2))

		# summing across each agent j to get the advantage
		# so we sum across the second last dimension which does A[t,j] = sum(V[t,i,j] - discounted_rewards[t,i])
		# NO MASKING OF ADVANTAGES
		# advantage = torch.sum(self.calculate_advantages(discounted_rewards, V_values_i_j, rewards, dones, True, False),dim=-2)
		
		
		# MASKING ADVANTAGES
		# Top 1
		masking_advantage = torch.transpose(F.one_hot(torch.argmax(weights_adv_i.squeeze(-1).detach(), dim=-1), num_classes=self.num_agents),-1,-2)
		# Top K
		# values, indices = torch.topk(weights,k=self.top_k,dim=-1)
		# masking_advantage = torch.transpose(torch.sum(F.one_hot(indices, num_classes=self.num_agents), dim=-2),-1,-2)
		# Above threshold
		# masking_advantage = torch.transpose((weights>self.select_above_threshold).int(),-1,-2)

		advantage = torch.sum(self.calculate_advantages(discounted_rewards, V_values_i_j, rewards, dones, True, False) * masking_advantage,dim=-2)

		# SCALING ADVANTAGES
		# advantage = torch.sum(self.calculate_advantages(discounted_rewards, V_values, rewards, dones, True, False) * masking_advantage *self.num_agents,dim=-2)
		
		# SOFT WEIGHTED ADVANTAGES
		# advantage = torch.sum(self.calculate_advantages(discounted_rewards, V_values, rewards, dones, True, False) * weights * self.num_agents ,dim=-2)

		probs = Categorical(probs)
		policy_loss = -probs.log_prob(actions) * advantage.detach()
		policy_loss = policy_loss.mean() - self.entropy_pen*entropy
		# policy_loss = torch.Tensor([0])
	# # ***********************************************************************************
		
	# **********************************
		self.critic_optimizer.zero_grad()
		value_i_j_loss.backward(retain_graph=False)
		grad_norm_value_i_j = torch.nn.utils.clip_grad_norm_(self.critic_network.parameters(),0.5)
		self.critic_optimizer.step()

		self.adv_optimizer.zero_grad()
		advantage_loss.backward(retain_graph=False)
		grad_norm_adv = torch.nn.utils.clip_grad_norm_(self.advantage_network.parameters(),0.5)
		self.adv_optimizer.step()

		self.value_optimizer.zero_grad()
		value_i_loss.backward(retain_graph=False)
		grad_norm_value_i = torch.nn.utils.clip_grad_norm_(self.value_network.parameters(),0.5)
		self.value_optimizer.step()


		self.policy_optimizer.zero_grad()
		policy_loss.backward(retain_graph=False)
		grad_norm_policy = torch.nn.utils.clip_grad_norm_(self.policy_network.parameters(),0.5)
		self.policy_optimizer.step()
		# grad_norm_policy = 0

		# V values
		return value_i_j_loss, value_i_loss, advantage_loss, policy_loss, entropy, grad_norm_value_i_j, grad_norm_value_i, grad_norm_adv, grad_norm_policy, weights_v_i_j, weights_v_i, weights_adv_i