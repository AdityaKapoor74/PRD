import numpy as np
import torch 
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
from torch.autograd import Variable
from torch.distributions import Categorical
from a2c_collision_avoidance import PolicyNetwork, ScalarDotProductCriticNetwork, ScalarDotProductPolicyNetwork
from a2c_paired_agents import ScalarDotProductCriticNetworkV5,ScalarDotProductCriticNetworkV3,ScalarDotProductPolicyNetworkV2
import torch.nn.functional as F

class A2CAgent:

	def __init__(
		self, 
		env, 
		dictionary
		):

		self.env = env
		self.value_lr = dictionary["value_lr"]
		self.policy_lr = dictionary["policy_lr"]
		self.gamma = dictionary["gamma"]
		self.entropy_pen = dictionary["entropy_pen"]
		self.trace_decay = dictionary["trace_decay"]
		self.top_k = dictionary["top_k"]
		self.l1_pen = dictionary["l1_pen"]
		self.anneal_l1_pen = dictionary["anneal_l1_pen"]
		if self.anneal_l1_pen:
			print("WHY?!?!?!?!")
			print('self.anneal_l1_pen: ', self.anneal_l1_pen)
			self.anneal_rate = dictionary["anneal_rate"]
		self.td_lambda = dictionary["td_lambda"]
		self.critic_loss_type = dictionary["critic_loss_type"]
		self.max_episodes = dictionary["max_episodes"]
		# Used for masking advantages above a threshold
		self.select_above_threshold = dictionary["select_above_threshold"]
		# cut the tail of softmax --> Used in softmax with normalization
		self.softmax_cut_threshold = dictionary["softmax_cut_threshold"]

		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		# self.device = "cpu"
		
		self.num_agents = self.env.n
		self.num_actions = self.env.action_space[0].n
		self.gif = dictionary["gif"]


		self.experiment_type = dictionary["experiment_type"]
		print('experiment_type: ', self.experiment_type)
		self.scaling_factor = None
		if self.experiment_type == "without_prd_scaled" or self.experiment_type == "with_prd_soft_adv_scaled":
			self.scaling_factor = self.num_agents
		elif "top" in self.experiment_type:
			self.scaling_factor = self.num_agents/self.top_k


		self.greedy_policy = torch.zeros(self.num_agents,self.num_agents).to(self.device)
		for i in range(self.num_agents):
			self.greedy_policy[i][i] = 1

		print(self.experiment_type, self.scaling_factor)


		# SCALAR DOT PRODUCT
		self.obs_input_dim = 2*3
		self.obs_act_input_dim = self.obs_input_dim + self.num_actions # (pose,vel,goal pose, paired agent goal pose) --> observations 
		self.obs_act_output_dim = dictionary["obs_act_output_dim"]# = 16
		self.final_input_dim = self.obs_act_output_dim #+ self.obs_input_dim #self.obs_z_output_dim + self.weight_input_dim
		self.final_output_dim = 1
		if dictionary["critic_version"] == 1:
			self.critic_network =   ScalarDotProductCriticNetwork(self.obs_act_input_dim, self.obs_act_output_dim, self.final_input_dim, self.final_output_dim, self.num_agents, self.num_actions, self.softmax_cut_threshold).to(self.device)
		elif dictionary["critic_version"] == 3:
			print("USING CRITIC VERSION 3!!!!!!!!!!")
			self.critic_network = ScalarDotProductCriticNetworkV3(self.obs_act_input_dim, self.obs_act_output_dim, self.final_input_dim, self.final_output_dim, self.num_agents, self.num_actions, self.softmax_cut_threshold).to(self.device)

		elif dictionary["critic_version"] == 5:
			print("USING CRITIC VERSION 5!!!!!!!!!!")
			self.critic_network = ScalarDotProductCriticNetworkV5(self.obs_act_input_dim, self.obs_act_output_dim, self.final_input_dim, self.final_output_dim, self.num_agents, self.num_actions, self.softmax_cut_threshold).to(self.device)

		# SCALAR DOT PRODUCT POLICY NETWORK
		self.obs_input_dim = 2*3
		self.obs_output_dim = dictionary["obs_act_output_dim"]
		self.final_input_dim = self.obs_output_dim
		self.final_output_dim = self.num_actions

		if dictionary["policy_version"] == 1:
			self.policy_network = ScalarDotProductPolicyNetwork(self.obs_input_dim, self.obs_output_dim, self.final_input_dim, self.final_output_dim, self.num_agents, self.num_actions, self.softmax_cut_threshold).to(self.device)
		elif dictionary["policy_version"] == 2:
			print('USING POLICY VERSION 2!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!1')
			self.policy_network = ScalarDotProductPolicyNetworkV2(self.obs_input_dim, self.obs_output_dim, self.final_input_dim, self.final_output_dim, self.num_agents, self.num_actions, self.softmax_cut_threshold).to(self.device)


		# MLP POLICY
		# self.policy_input_dim = 2*(3+2*(self.num_agents-1)) #2 for pose, 2 for vel and 2 for goal of current agent and rest (2 each) for relative position and relative velocity of other agents
		# self.policy_output_dim = self.env.action_space[0].n
		# policy_network_size = (self.policy_input_dim,512,256,self.policy_output_dim)
		# self.policy_network = PolicyNetwork(policy_network_size).to(self.device)


		# Loading models
		# model_path_value = "../../../models/Scalar_dot_product/collision_avoidance/4_Agents/SingleAttentionMechanism/with_prd_soft_adv/critic_networks/14-05-2021VN_ATN_FCN_lr0.01_PN_FCN_lr0.0002_GradNorm0.5_Entropy0.008_trace_decay0.98lambda_0.001topK_2select_above_threshold0.1softmax_cut_threshold0.1_epsiode29000.pt"
		# model_path_policy = "../../../models/Scalar_dot_product/collision_avoidance/4_Agents/SingleAttentionMechanism/with_prd_soft_adv/actor_networks/14-05-2021_PN_FCN_lr0.0002VN_SAT_FCN_lr0.01_GradNorm0.5_Entropy0.008_trace_decay0.98lambda_0.001topK_2select_above_threshold0.1softmax_cut_threshold0.1_epsiode29000.pt"
		# For CPU
		# self.critic_network.load_state_dict(torch.load(model_path_value,map_location=torch.device('cpu')))
		# self.policy_network.load_state_dict(torch.load(model_path_policy,map_location=torch.device('cpu')))
		# # For GPU
		# self.critic_network.load_state_dict(torch.load(model_path_value))
		# self.policy_network.load_state_dict(torch.load(model_path_policy))


		self.critic_optimizer = optim.Adam(self.critic_network.parameters(),lr=self.value_lr)
		self.policy_optimizer = optim.Adam(self.policy_network.parameters(),lr=self.policy_lr)


	def get_action(self,state):
		# MLP
		# state = torch.FloatTensor(state).to(self.device)
		# dists = self.policy_network.forward(state)
		# probs = Categorical(dists)
		# index = probs.sample().cpu().detach().item()

		# return index

		# GNN
		state = torch.FloatTensor([state]).to(self.device)
		dists, _ = self.policy_network.forward(state)
		index = [Categorical(dist).sample().cpu().detach().item() for dist in dists[0]]
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
		
		
		

	def calculate_deltas(self, values, rewards, dones):
		target_values = []
		next_value = 0
		rewards = rewards.unsqueeze(-1)
		dones = dones.unsqueeze(-1)
		masks = 1-dones
		for t in reversed(range(0, len(rewards))):
			value_target = rewards[t] + (self.gamma * next_value * masks[t]) - values.data[t]
			next_value = values.data[t]
			target_values.insert(0,value_target)
		target_values = torch.stack(target_values)

		return target_values

	def nstep_returns(self,values, rewards, dones):
		target_values = self.calculate_deltas(values, rewards, dones)
		target_Vs = self.calculate_returns(target_values, self.gamma*self.td_lambda) + values.data
		return target_Vs



	def update(self,states_critic,next_states_critic,one_hot_actions,one_hot_next_actions,actions,states_actor,next_states_actor,rewards,dones,episode=None):

		'''
		Getting the probability mass function over the action space for each agent
		'''
		# MLP
		# probs = self.policy_network.forward(actor_graphs).reshape(-1,self.num_agents,self.num_actions)
		# probs = self.policy_network.forward(states_actor)
		# next_probs = self.policy_network.forward(next_states_actor)

		# GNN
		probs, weight_policy = self.policy_network.forward(states_actor)

		'''
		Calculate V values
		'''
		V_values, weights = self.critic_network.forward(states_critic, probs.detach(), one_hot_actions)
		# V_values_next, _ = self.critic_network.forward(next_states_critic, next_probs.detach(), one_hot_next_actions)
		V_values = V_values.reshape(-1,self.num_agents,self.num_agents)
		# V_values_next = V_values_next.reshape(-1,self.num_agents,self.num_agents)

		
	# # ***********************************************************************************
	# 	#update critic (value_net)
	# we need a TxNxN vector so inflate the discounted rewards by N --> cloning the discounted rewards for an agent N times
		discounted_rewards = self.calculate_returns(rewards,self.gamma).unsqueeze(-2).repeat(1,self.num_agents,1).to(self.device)
		discounted_rewards = torch.transpose(discounted_rewards,-1,-2)

		# BOOTSTRAP LOSS
		# target_values = torch.transpose(rewards.unsqueeze(-2).repeat(1,self.num_agents,1),-1,-2) + self.gamma*V_values_next*(1-dones.unsqueeze(-1))
		# value_loss = F.smooth_l1_loss(V_values,target_values)

		# MONTE CARLO LOSS
		# print('weights: ', weights)
		weights_off_diagonal = weights * (1 - torch.eye(self.num_agents,device=self.device))
		# print('weights_off_diagonal: ', weights_off_diagonal)
		l1_weights = torch.mean(weights_off_diagonal)
		if not self.anneal_l1_pen:
			if self.critic_loss_type == 'monte_carlo':
				value_loss = F.smooth_l1_loss(V_values,discounted_rewards) + self.l1_pen*l1_weights
			elif self.critic_loss_type == 'td_lambda':
				V_values_target = self.nstep_returns(V_values, rewards, dones)
				value_loss = F.smooth_l1_loss(V_values,V_values_target.detach())
			else:
				assert False
		else:
			assert episode is not None
			l1_pen = self.l1_pen * np.exp(-self.anneal_rate*episode)
			value_loss = F.smooth_l1_loss(V_values,discounted_rewards) + l1_pen*l1_weights

		# REGRESSING ON IMMEDIATE REWARD
		# value_loss = F.smooth_l1_loss(V_values,torch.transpose(rewards.unsqueeze(-2).repeat(1,self.num_agents,1),-1,-2))
		
		# # ***********************************************************************************
	# 	#update actor (policy net)
	# # ***********************************************************************************
		entropy = -torch.mean(torch.sum(probs * torch.log(torch.clamp(probs, 1e-10,1.0)), dim=2))

		# summing across each agent j to get the advantage
		# so we sum across the second last dimension which does A[t,j] = sum(V[t,i,j] - discounted_rewards[t,i])
		advantage = None
		if self.experiment_type == "without_prd" or self.experiment_type == "without_prd_scaled":
			advantage = torch.sum(self.calculate_advantages(discounted_rewards, V_values, rewards, dones, True, False),dim=-2)
		elif "top" == self.experiment_type:
			values, indices = torch.topk(weights,k=self.top_k,dim=-1)
			masking_advantage = torch.transpose(torch.sum(F.one_hot(indices, num_classes=self.num_agents), dim=-2),-1,-2)
			advantage = torch.sum(self.calculate_advantages(discounted_rewards, V_values, rewards, dones, True, False) * masking_advantage,dim=-2)
		elif self.experiment_type in "above_threshold":
			masking_advantage = torch.transpose((weights>self.select_above_threshold).int(),-1,-2)
			advantage = torch.sum(self.calculate_advantages(discounted_rewards, V_values, rewards, dones, True, False) * masking_advantage,dim=-2)
		elif self.experiment_type == "with_prd_soft_adv" or self.experiment_type == "with_prd_soft_adv_scaled":
			advantage = torch.sum(self.calculate_advantages(discounted_rewards, V_values, rewards, dones, True, False) * weights.transpose(-1,-2) ,dim=-2)
		elif self.experiment_type == "greedy_policy":
			
			advantage = torch.sum(self.calculate_advantages(discounted_rewards, V_values, rewards, dones, True, False) * self.greedy_policy ,dim=-2)
		elif self.experiment_type == "greedy_and_top":
			values, indices = torch.topk(weights,k=self.top_k,dim=-1)
			masking_advantage = torch.transpose(torch.sum(F.one_hot(indices, num_classes=self.num_agents), dim=-2),-1,-2)
			masking_advantage = masking_advantage + self.greedy_policy > 0
			advantage = torch.sum(self.calculate_advantages(discounted_rewards, V_values, rewards, dones, True, False) * masking_advantage ,dim=-2)
		elif self.experiment_type == "without_prd_mean":
			advantage = 1/self.num_agents*torch.sum(self.calculate_advantages(discounted_rewards, V_values, rewards, dones, True, False),dim=-2)



		else:
			print('Unknown experiment type')
			assert False


		if "scaled" in self.experiment_type:
			advantage = advantage * self.scaling_factor


		probs = Categorical(probs)
		policy_loss = -probs.log_prob(actions) * advantage.detach()
		policy_loss = policy_loss.mean() - self.entropy_pen*entropy
		# policy_loss = torch.Tensor([0])
	# # ***********************************************************************************
		
	# **********************************
		self.critic_optimizer.zero_grad()
		value_loss.backward(retain_graph=False)
		grad_norm_value = torch.nn.utils.clip_grad_norm_(self.critic_network.parameters(),0.5)
		self.critic_optimizer.step()


		self.policy_optimizer.zero_grad()
		policy_loss.backward(retain_graph=False)
		grad_norm_policy = torch.nn.utils.clip_grad_norm_(self.policy_network.parameters(),0.5)
		self.policy_optimizer.step()
		# grad_norm_policy = 0

		# V values
		return value_loss,policy_loss,entropy,grad_norm_value,grad_norm_policy,weights, weight_policy