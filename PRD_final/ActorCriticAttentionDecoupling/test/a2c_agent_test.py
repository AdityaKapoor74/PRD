import numpy as np
import torch 
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
from torch.autograd import Variable
from torch.distributions import Categorical
from a2c_test import *
import torch.nn.functional as F

class A2CAgent:

	def __init__(
		self, 
		env, 
		dictionary
		):

		self.env = env
		self.env_name = dictionary["env"]
		self.value_lr = dictionary["value_lr"]
		self.policy_lr = dictionary["policy_lr"]
		self.gamma = dictionary["gamma"]
		self.entropy_pen = dictionary["entropy_pen"]
		self.trace_decay = dictionary["trace_decay"]
		self.top_k = dictionary["top_k"]
		self.critic_type = dictionary["critic_type"]
		self.gae = dictionary["gae"]
		self.norm_adv = dictionary["norm_adv"]
		self.norm_rew = dictionary["norm_rew"]
		# Used for masking advantages above a threshold
		self.select_above_threshold = dictionary["select_above_threshold"]
		# cut the tail of softmax --> Used in softmax with normalization
		self.softmax_cut_threshold = dictionary["softmax_cut_threshold"]
		self.attention_heads = dictionary["attention_heads"]
		self.freeze_policy = dictionary["freeze_policy"]
		self.episode_counter = 0
		self.l1_pen = dictionary["l1_pen"]

		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		# self.device = "cpu"
		
		self.num_agents = self.env.n
		self.num_actions = self.env.action_space[0].n
		self.gif = dictionary["gif"]

		self.experiment_type = dictionary["experiment_type"]

		self.greedy_policy = torch.zeros(self.num_agents,self.num_agents).to(self.device)
		for i in range(self.num_agents):
			self.greedy_policy[i][i] = 1

		print("CRITIC TYPE", self.critic_type)
		print("EXPERIMENT TYPE", self.experiment_type)

		# TD lambda
		self.lambda_ = 0.8

		# PAIRED AGENT
		if self.env_name == "paired_by_sharing_goals":
			obs_dim = 2*4
		elif self.env_name in ["multi_circular", "collision_avoidance", "collision_avoidance_no_width", "reach_landmark_social_dilemma"]:
			obs_dim = 2*3
		elif self.env_name == "color_social_dilemma":
			obs_dim = 2*2 + 1 + self.num_agents*3

		# MLP_CRITIC_STATE, MLP_CRITIC_STATE_ACTION, GNN_CRITIC_STATE, GNN_CRITIC_STATE_ACTION
		if self.critic_type == "MLP_CRITIC_STATE":
			self.critic_network = StateOnlyMLPCritic(obs_dim, self.num_agents).to(self.device)
		elif self.critic_type == "MLP_CRITIC_STATE_ACTION":
			self.critic_network = StateActionMLPCritic(obs_dim, self.num_actions, self.num_agents).to(self.device)
		elif self.critic_type == "GNN_CRITIC_STATE":
			self.critic_network = StateOnlyGATCritic(obs_dim, 128, 128, 1, self.num_agents, self.num_actions).to(self.device)
		elif self.critic_type == "GNN_CRITIC_STATE_ACTION":
			self.critic_network = StateActionGATCritic(obs_dim, 128, obs_dim+self.num_actions, 128, 128, 1, self.num_agents, self.num_actions).to(self.device)
		elif self.critic_type == "ALL" or self.critic_type == "ALL_W_POL":
			self.critic_network_1 = StateOnlyMLPCritic(obs_dim, self.num_agents).to(self.device)
			self.critic_network_2 = StateActionMLPCritic(obs_dim, self.num_actions, self.num_agents).to(self.device)
			self.critic_network_3 = StateOnlyGATCritic(obs_dim, 128, 128, 1, self.num_agents, self.num_actions).to(self.device)
			self.critic_network_4 = StateActionGATCritic(obs_dim, 128, obs_dim+self.num_actions, 128, 128, 1, self.num_agents, self.num_actions).to(self.device)
		elif self.critic_type == "MultiHead":
			self.critic_network = StateActionGATCriticMultiHead(obs_dim, 128, obs_dim+self.num_actions, 128//self.attention_heads, 128, 1, self.num_agents, self.num_actions, attention_heads=self.attention_heads).to(self.device)
		elif self.critic_type == "AttentionCriticV1":
			self.critic_network = AttentionCriticV1(obs_dim, 128, obs_dim+self.num_actions, 128, 128, 1, self.num_agents, self.num_actions, attend_heads=self.attention_heads).to(self.device)
		elif self.critic_type == "NonResV1":
			self.critic_network = StateActionGATCriticWoResConnV1(obs_dim, 128, obs_dim+self.num_actions, 128, 128, 1, self.num_agents, self.num_actions).to(self.device)
		elif self.critic_type == "ResV1":
			self.critic_network = StateActionGATCriticWResConnV1(obs_dim, 128, obs_dim+self.num_actions, 128, 128, 1, self.num_agents, self.num_actions).to(self.device)
		elif self.critic_type == "NonResV2":
			self.critic_network = StateActionGATCriticWoResConnV2(obs_dim, 128, obs_dim+self.num_actions, 128, 128, 1, self.num_agents, self.num_actions).to(self.device)
		elif self.critic_type == "ResV2":
			self.critic_network = StateActionGATCriticWResConnV2(obs_dim, 128, obs_dim+self.num_actions, 128, 128, 1, self.num_agents, self.num_actions).to(self.device)
		elif self.critic_type == "NonResV3":
			self.critic_network = StateActionGATCriticWoResConnV3(obs_dim, 128, obs_dim+self.num_actions, 128, 128, 1, self.num_agents, self.num_actions).to(self.device)
		elif self.critic_type == "ResV3":
			self.critic_network = StateActionGATCriticWResConnV3(obs_dim, 128, obs_dim+self.num_actions, 128, 128, 1, self.num_agents, self.num_actions).to(self.device)
		elif self.critic_type == "MLPToGNN":
			self.critic_network_1 = MLPToGNNV1(obs_dim,self.num_agents).to(self.device)
			self.critic_network_2 = MLPToGNNV2(obs_dim,self.num_agents).to(self.device)
			self.critic_network_3 = MLPToGNNV3(obs_dim,self.num_agents).to(self.device)
			self.critic_network_4 = MLPToGNNV4(obs_dim,self.num_actions,self.num_agents).to(self.device)
			self.critic_network_5 = MLPToGNNV5(obs_dim, 128, obs_dim+self.num_actions, 128, 128, 1, self.num_agents, self.num_actions).to(self.device)
			self.critic_network_6 = MLPToGNNV6(obs_dim, 128, obs_dim+self.num_actions, 128, 128, 1, self.num_agents, self.num_actions).to(self.device)
			self.critic_network_7 = MLPToGNNV7(obs_dim, 128, obs_dim+self.num_actions, 128, 128, 1, self.num_agents, self.num_actions).to(self.device)
			self.critic_network_8 = MLPToGNNV8(obs_dim, 128, obs_dim+self.num_actions, 128, 128, 1, self.num_agents, self.num_actions).to(self.device)
		elif self.critic_type == "MLPToGNNV5":
			self.critic_network = MLPToGNNV5(obs_dim, 128, obs_dim+self.num_actions, 128, 128, 1, self.num_agents, self.num_actions).to(self.device)
		elif self.critic_type == "MLPToGNNV6":
			self.critic_network = MLPToGNNV6(obs_dim, 128, obs_dim+self.num_actions, 128, 128, 1, self.num_agents, self.num_actions).to(self.device)
		elif self.critic_type == "GNNTanhRelU":
			self.critic_network = GNNTanhRelU(obs_dim, 128, obs_dim+self.num_actions, 128, 128, 1, self.num_agents, self.num_actions).to(self.device)
		elif self.critic_type == "DualMLPGATCritic_MLPTrain":
			self.critic_network_1 = MLPToGNNV1(obs_dim,self.num_agents).to(self.device)
			self.critic_network_2 = MLPToGNNV6(obs_dim, 128, obs_dim+self.num_actions, 128, 128, 1, self.num_agents, self.num_actions).to(self.device)
		elif self.critic_type == "DualGATGATCritic":
			self.critic_network_1 = MLPToGNNV6(obs_dim, 128, obs_dim+self.num_actions, 128, 128, 1, self.num_agents, self.num_actions).to(self.device)
			self.critic_network_2 = MLPToGNNV6(obs_dim, 128, obs_dim+self.num_actions, 128, 128, 1, self.num_agents, self.num_actions).to(self.device)
		elif self.critic_type == "GATPushBall":
			obs_agent_input_dim = obs_ball_input_dim = 2*2
			obs_act_input_dim = obs_agent_input_dim + self.num_actions
			obs_agent_output_dim = obs_ball_output_dim = obs_act_output_dim = 128
			final_input_dim = obs_ball_output_dim + obs_act_output_dim
			final_output_dim = 1
			self.critic_network = GATPushBall(obs_agent_input_dim, obs_agent_output_dim, obs_ball_input_dim, obs_ball_output_dim, obs_act_input_dim, obs_act_output_dim, final_input_dim, final_output_dim, self.num_agents, self.num_actions, threshold=0.1).to(self.device)




		# MLP POLICY
		if self.env_name in ["paired_by_sharing_goals","multi_circular", "collision_avoidance", "collision_avoidance_no_width", "reach_landmark_social_dilemma"]:
			obs_dim = 2*3
		elif self.env_name == "color_social_dilemma":
			obs_dim = 2*2 + 1 + self.num_agents*3
		self.policy_network = MLPPolicyNetwork(obs_dim, self.num_agents, self.num_actions).to(self.device)


		# Loading models
		# model_path_value = "../../../../tests/color_social_dilemma/models/color_social_dilemma_greedy_policy_MLPToGNNV6_withMLPPol_with_l1_pen/critic_networks/02-07-2021VN_ATN_FCN_lr0.001_PN_ATN_FCN_lr0.0005_GradNorm0.5_Entropy0.008_trace_decay0.98topK_0select_above_threshold0.1softmax_cut_threshold0.1_epsiode100000_MLPToGNNV6.pt"
		# model_path_policy = "../../../../tests/color_social_dilemma/models/color_social_dilemma_greedy_policy_MLPToGNNV6_withMLPPol_with_l1_pen/actor_networks/02-07-2021_PN_ATN_FCN_lr0.0005VN_SAT_FCN_lr0.001_GradNorm0.5_Entropy0.008_trace_decay0.98topK_0select_above_threshold0.1softmax_cut_threshold0.1_epsiode100000_MLPToGNNV6.pt"
		# For CPU
		# self.critic_network.load_state_dict(torch.load(model_path_value,map_location=torch.device('cpu')))
		# self.policy_network.load_state_dict(torch.load(model_path_policy,map_location=torch.device('cpu')))
		# # For GPU
		# self.critic_network.load_state_dict(torch.load(model_path_value))
		# self.policy_network.load_state_dict(torch.load(model_path_policy))

		if self.critic_type == "ALL":
			self.critic_optimizer_1 = optim.Adam(self.critic_network_1.parameters(),lr=self.value_lr[0])
			self.critic_optimizer_2 = optim.Adam(self.critic_network_2.parameters(),lr=self.value_lr[1])
			self.critic_optimizer_3 = optim.Adam(self.critic_network_3.parameters(),lr=self.value_lr[2])
			self.critic_optimizer_4 = optim.Adam(self.critic_network_4.parameters(),lr=self.value_lr[3])
		elif self.critic_type == "ALL_W_POL":
			self.critic_optimizer_1 = optim.Adam(self.critic_network_1.parameters(),lr=self.value_lr[0])
			self.critic_optimizer_2 = optim.Adam(self.critic_network_2.parameters(),lr=self.value_lr[1])
			self.critic_optimizer_3 = optim.Adam(self.critic_network_3.parameters(),lr=self.value_lr[2])
			self.critic_optimizer_4 = optim.Adam(self.critic_network_4.parameters(),lr=self.value_lr[3])
			self.policy_optimizer = optim.Adam(self.policy_network.parameters(),lr=self.policy_lr)
		elif self.critic_type == "MLPToGNN":
			self.critic_optimizer_1 = optim.Adam(self.critic_network_1.parameters(),lr=self.value_lr[0])
			self.critic_optimizer_2 = optim.Adam(self.critic_network_2.parameters(),lr=self.value_lr[1])
			self.critic_optimizer_3 = optim.Adam(self.critic_network_3.parameters(),lr=self.value_lr[2])
			self.critic_optimizer_4 = optim.Adam(self.critic_network_4.parameters(),lr=self.value_lr[3])
			self.critic_optimizer_5 = optim.Adam(self.critic_network_5.parameters(),lr=self.value_lr[4])
			self.critic_optimizer_6 = optim.Adam(self.critic_network_6.parameters(),lr=self.value_lr[5])
			self.critic_optimizer_7 = optim.Adam(self.critic_network_7.parameters(),lr=self.value_lr[6])
			self.critic_optimizer_8 = optim.Adam(self.critic_network_8.parameters(),lr=self.value_lr[7])
			self.policy_optimizer = optim.Adam(self.policy_network.parameters(),lr=self.policy_lr)
		elif "Dual" in self.critic_type:
			self.critic_optimizer_1 = optim.Adam(self.critic_network_1.parameters(),lr=self.value_lr[0])
			self.critic_optimizer_2 = optim.Adam(self.critic_network_2.parameters(),lr=self.value_lr[1])
			self.policy_optimizer = optim.Adam(self.policy_network.parameters(),lr=self.policy_lr)
		else:
			self.critic_optimizer = optim.Adam(self.critic_network.parameters(),lr=self.value_lr)
			self.policy_optimizer = optim.Adam(self.policy_network.parameters(),lr=self.policy_lr)


	def get_action(self,state):
		state = torch.FloatTensor([state]).to(self.device)
		dists, _ = self.policy_network.forward(state)
		index = [Categorical(dist).sample().cpu().detach().item() for dist in dists[0]]
		return index

	def get_action_push_ball(self,state_agent, state_ball):
		state = torch.FloatTensor([state]).to(self.device)
		dists, _ = self.policy_network.forward(state)
		index = [Categorical(dist).sample().cpu().detach().item() for dist in dists[0]]
		return index



	def calculate_advantages(self,returns, values, rewards, dones):
		
		advantages = None

		if self.gae:
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
		
		if self.norm_adv:
			
			advantages = (advantages - advantages.mean()) / advantages.std()
		
		return advantages


	def calculate_deltas(self, values, rewards, dones):
		deltas = []
		next_value = 0
		rewards = rewards.unsqueeze(-1)
		dones = dones.unsqueeze(-1)
		masks = 1-dones
		for t in reversed(range(0, len(rewards))):
			td_error = rewards[t] + (self.gamma * next_value * masks[t]) - values.data[t]
			next_value = values.data[t]
			deltas.insert(0,td_error)
		deltas = torch.stack(deltas)

		return deltas


	def nstep_returns(self,values, rewards, dones):
		deltas = self.calculate_deltas(values, rewards, dones)
		advs = self.calculate_returns(deltas, self.gamma*self.lambda_)
		target_Vs = advs+values
		return target_Vs


	def calculate_returns(self,rewards, discount_factor):
		returns = []
		R = 0
		
		for r in reversed(rewards):
			R = r + R * discount_factor
			returns.insert(0, R)
		
		returns_tensor = torch.stack(returns).to(self.device)
		
		if self.norm_rew:
			
			returns_tensor = (returns_tensor - returns_tensor.mean()) / returns_tensor.std()
			
		return returns_tensor
		
		


	def update(self,states_critic,next_states_critic,one_hot_actions,one_hot_next_actions,actions,states_actor,next_states_actor,rewards,dones):

		'''
		Getting the probability mass function over the action space for each agent
		'''
		probs, weight_policy = self.policy_network.forward(states_actor)

		'''
		Calculate V values
		'''
		if self.critic_type == "ALL":
			value_loss = []
			grad_norm_value = []
			weights = []
			for i in range(1,5):
				if i == 1:
					V_values, weights_ = self.critic_network_1.forward(states_critic, probs.detach(), one_hot_actions)
					# V_values_next, _ = self.critic_network_1.forward(next_states_critic, next_probs.detach(), one_hot_next_actions)
				elif i == 2:
					V_values, weights_ = self.critic_network_2.forward(states_critic, probs.detach(), one_hot_actions)
					# V_values_next, _ = self.critic_network_2.forward(next_states_critic, next_probs.detach(), one_hot_next_actions)
				elif i == 3:
					V_values, weights_ = self.critic_network_3.forward(states_critic, probs.detach(), one_hot_actions)
					# V_values_next, _ = self.critic_network_3.forward(next_states_critic, next_probs.detach(), one_hot_next_actions)
				elif i == 4:
					V_values, weights_ = self.critic_network_4.forward(states_critic, probs.detach(), one_hot_actions)
					# V_values_next, _ = self.critic_network_4.forward(next_states_critic, next_probs.detach(), one_hot_next_actions)
				


				V_values = V_values.reshape(-1,self.num_agents,self.num_agents)
				# V_values_next = V_values_next.reshape(-1,self.num_agents,self.num_agents)

			
				# # ***********************************************************************************
				# update critic (value_net)
				# we need a TxNxN vector so inflate the discounted rewards by N --> cloning the discounted rewards for an agent N times
				# discounted_rewards = self.calculate_returns(rewards,self.gamma).unsqueeze(-2).repeat(1,self.num_agents,1).to(self.device)
				# discounted_rewards = torch.transpose(discounted_rewards,-1,-2)

				# BOOTSTRAP LOSS
				# target_values = torch.transpose(rewards.unsqueeze(-2).repeat(1,self.num_agents,1),-1,-2) + self.gamma*V_values_next*(1-dones.unsqueeze(-1))
				# value_loss = F.smooth_l1_loss(V_values,target_values)

				# MONTE CARLO LOSS
				# value_loss = F.smooth_l1_loss(V_values,discounted_rewards)

				# TD lambda 
				Value_target_ = self.nstep_returns(V_values, rewards, dones).detach()
				value_loss_ = F.smooth_l1_loss(V_values, Value_target_)

				if i == 1:
					self.critic_optimizer_1.zero_grad()
					value_loss_.backward(retain_graph=False)
					grad_norm_value_ = torch.nn.utils.clip_grad_norm_(self.critic_network_1.parameters(),100.0)
					self.critic_optimizer_1.step()
				elif i == 2:
					self.critic_optimizer_2.zero_grad()
					value_loss_.backward(retain_graph=False)
					grad_norm_value_ = torch.nn.utils.clip_grad_norm_(self.critic_network_2.parameters(),100.0)
					self.critic_optimizer_2.step()
				elif i == 3:
					self.critic_optimizer_3.zero_grad()
					value_loss_.backward(retain_graph=False)
					grad_norm_value_ = torch.nn.utils.clip_grad_norm_(self.critic_network_3.parameters(),100.0)
					self.critic_optimizer_3.step()
				elif i == 4:
					self.critic_optimizer_4.zero_grad()
					value_loss_.backward(retain_graph=False)
					grad_norm_value_ = torch.nn.utils.clip_grad_norm_(self.critic_network_4.parameters(),100.0)
					self.critic_optimizer_4.step()

				value_loss.append(value_loss_)
				grad_norm_value.append(grad_norm_value_)
				weights.append(weights_)

			return value_loss,None,None,grad_norm_value,None,weights,None

		elif self.critic_type == "ALL_W_POL":
			value_loss = []
			grad_norm_value = []
			weights = []
			for i in range(1,5):
				if i == 1:
					V_values, weights_ = self.critic_network_1.forward(states_critic, probs.detach(), one_hot_actions)
					# V_values_next, _ = self.critic_network_1.forward(next_states_critic, next_probs.detach(), one_hot_next_actions)
				elif i == 2:
					V_values, weights_ = self.critic_network_2.forward(states_critic, probs.detach(), one_hot_actions)
					# V_values_next, _ = self.critic_network_2.forward(next_states_critic, next_probs.detach(), one_hot_next_actions)
				elif i == 3:
					V_values, weights_ = self.critic_network_3.forward(states_critic, probs.detach(), one_hot_actions)
					# V_values_next, _ = self.critic_network_3.forward(next_states_critic, next_probs.detach(), one_hot_next_actions)
				elif i == 4:
					V_values, weights_ = self.critic_network_4.forward(states_critic, probs.detach(), one_hot_actions)
					# V_values_next, _ = self.critic_network_4.forward(next_states_critic, next_probs.detach(), one_hot_next_actions)
				


				V_values = V_values.reshape(-1,self.num_agents,self.num_agents)
				# V_values_next = V_values_next.reshape(-1,self.num_agents,self.num_agents)

			
				# # ***********************************************************************************
				# update critic (value_net)
				# we need a TxNxN vector so inflate the discounted rewards by N --> cloning the discounted rewards for an agent N times
				if not(self.gae):
					discounted_rewards = self.calculate_returns(rewards,self.gamma).unsqueeze(-2).repeat(1,self.num_agents,1).to(self.device)
					discounted_rewards = torch.transpose(discounted_rewards,-1,-2)
				else:
					discounted_rewards = None

				# BOOTSTRAP LOSS
				# target_values = torch.transpose(rewards.unsqueeze(-2).repeat(1,self.num_agents,1),-1,-2) + self.gamma*V_values_next*(1-dones.unsqueeze(-1))
				# value_loss = F.smooth_l1_loss(V_values,target_values)

				# MONTE CARLO LOSS
				# value_loss = F.smooth_l1_loss(V_values,discounted_rewards)

				# TD lambda 
				Value_target_ = self.nstep_returns(V_values, rewards, dones).detach()
				value_loss_ = F.smooth_l1_loss(V_values, Value_target_)

				if i == 1:
					self.critic_optimizer_1.zero_grad()
					value_loss_.backward(retain_graph=False)
					grad_norm_value_ = torch.nn.utils.clip_grad_norm_(self.critic_network_1.parameters(),0.5)
					self.critic_optimizer_1.step()
				elif i == 2:
					self.critic_optimizer_2.zero_grad()
					value_loss_.backward(retain_graph=False)
					grad_norm_value_ = torch.nn.utils.clip_grad_norm_(self.critic_network_2.parameters(),0.5)
					self.critic_optimizer_2.step()
				elif i == 3:
					self.critic_optimizer_3.zero_grad()
					value_loss_.backward(retain_graph=False)
					grad_norm_value_ = torch.nn.utils.clip_grad_norm_(self.critic_network_3.parameters(),0.5)
					self.critic_optimizer_3.step()
				elif i == 4:
					self.critic_optimizer_4.zero_grad()
					value_loss_.backward(retain_graph=False)
					grad_norm_value_ = torch.nn.utils.clip_grad_norm_(self.critic_network_4.parameters(),0.5)
					self.critic_optimizer_4.step()

				value_loss.append(value_loss_)
				grad_norm_value.append(grad_norm_value_)
				weights.append(weights_)

			# # ***********************************************************************************
			# update actor (policy net)
			# # ***********************************************************************************
			entropy = -torch.mean(torch.sum(probs * torch.log(torch.clamp(probs, 1e-10,1.0)), dim=2))

			# summing across each agent j to get the advantage
			# so we sum across the second last dimension which does A[t,j] = sum(V[t,i,j] - discounted_rewards[t,i])
			advantage = None
			V_values, weights_ = self.critic_network_1.forward(states_critic, probs.detach(), one_hot_actions)
			V_values = V_values.reshape(-1,self.num_agents,self.num_agents)
			if self.experiment_type == "without_prd":
				advantage = torch.sum(self.calculate_advantages(discounted_rewards, V_values, rewards, dones),dim=-2)
			elif "top" in self.experiment_type:
				values, indices = torch.topk(weights,k=self.top_k,dim=-1)
				masking_advantage = torch.transpose(torch.sum(F.one_hot(indices, num_classes=self.num_agents), dim=-2),-1,-2)
				advantage = torch.sum(self.calculate_advantages(discounted_rewards, V_values, rewards, dones) * masking_advantage,dim=-2)
			elif self.experiment_type in "above_threshold":
				masking_advantage = torch.transpose((weights>self.select_above_threshold).int(),-1,-2)
				advantage = torch.sum(self.calculate_advantages(discounted_rewards, V_values, rewards, dones) * masking_advantage,dim=-2)
			elif self.experiment_type == "with_prd_soft_adv":
				advantage = torch.sum(self.calculate_advantages(discounted_rewards, V_values, rewards, dones) * weights ,dim=-2)
			elif self.experiment_type == "greedy_policy":
				advantage = torch.sum(self.calculate_advantages(discounted_rewards, V_values, rewards, dones) * self.greedy_policy ,dim=-2)

		
			probs = Categorical(probs)
			policy_loss = -probs.log_prob(actions) * advantage.detach()
			policy_loss = policy_loss.mean() - self.entropy_pen*entropy

			self.policy_optimizer.zero_grad()
			policy_loss.backward(retain_graph=False)
			grad_norm_policy = torch.nn.utils.clip_grad_norm_(self.policy_network.parameters(),0.5)
			self.policy_optimizer.step()


			return value_loss,policy_loss,entropy,grad_norm_value,grad_norm_policy,weights,weight_policy


		elif self.critic_type == "MLPToGNN":
			value_loss = []
			grad_norm_value = []
			weights = []
			for i in range(1,9):
				if i == 1:
					V_values, weights_ = self.critic_network_1.forward(states_critic, probs.detach(), one_hot_actions)
					# V_values_next, _ = self.critic_network_1.forward(next_states_critic, next_probs.detach(), one_hot_next_actions)
				elif i == 2:
					V_values, weights_ = self.critic_network_2.forward(states_critic, probs.detach(), one_hot_actions)
					# V_values_next, _ = self.critic_network_2.forward(next_states_critic, next_probs.detach(), one_hot_next_actions)
				elif i == 3:
					V_values, weights_ = self.critic_network_3.forward(states_critic, probs.detach(), one_hot_actions)
					# V_values_next, _ = self.critic_network_3.forward(next_states_critic, next_probs.detach(), one_hot_next_actions)
				elif i == 4:
					V_values, weights_ = self.critic_network_4.forward(states_critic, probs.detach(), one_hot_actions)
					# V_values_next, _ = self.critic_network_4.forward(next_states_critic, next_probs.detach(), one_hot_next_actions)
				elif i == 5:
					V_values, weights_ = self.critic_network_5.forward(states_critic, probs.detach(), one_hot_actions)
					# V_values_next, _ = self.critic_network_4.forward(next_states_critic, next_probs.detach(), one_hot_next_actions)
				elif i == 6:
					V_values, weights_ = self.critic_network_6.forward(states_critic, probs.detach(), one_hot_actions)
					# V_values_next, _ = self.critic_network_4.forward(next_states_critic, next_probs.detach(), one_hot_next_actions)
				elif i == 7:
					V_values, weights_ = self.critic_network_7.forward(states_critic, probs.detach(), one_hot_actions)
					# V_values_next, _ = self.critic_network_4.forward(next_states_critic, next_probs.detach(), one_hot_next_actions)
				elif i == 8:
					V_values, _weights_, weights_ = self.critic_network_8.forward(states_critic, probs.detach(), one_hot_actions)
					# V_values_next, _ = self.critic_network_4.forward(next_states_critic, next_probs.detach(), one_hot_next_actions)
				


				V_values = V_values.reshape(-1,self.num_agents,self.num_agents)
				# V_values_next = V_values_next.reshape(-1,self.num_agents,self.num_agents)

			
				# # ***********************************************************************************
				# update critic (value_net)
				# we need a TxNxN vector so inflate the discounted rewards by N --> cloning the discounted rewards for an agent N times
				if not(self.gae):
					discounted_rewards = self.calculate_returns(rewards,self.gamma).unsqueeze(-2).repeat(1,self.num_agents,1).to(self.device)
					discounted_rewards = torch.transpose(discounted_rewards,-1,-2)
				else:
					discounted_rewards = None

				# BOOTSTRAP LOSS
				# target_values = torch.transpose(rewards.unsqueeze(-2).repeat(1,self.num_agents,1),-1,-2) + self.gamma*V_values_next*(1-dones.unsqueeze(-1))
				# value_loss = F.smooth_l1_loss(V_values,target_values)

				# MONTE CARLO LOSS
				# value_loss = F.smooth_l1_loss(V_values,discounted_rewards)

				# TD lambda 
				Value_target_ = self.nstep_returns(V_values, rewards, dones).detach()
				value_loss_ = F.smooth_l1_loss(V_values, Value_target_)

				if i == 1:
					self.critic_optimizer_1.zero_grad()
					value_loss_.backward(retain_graph=False)
					grad_norm_value_ = torch.nn.utils.clip_grad_norm_(self.critic_network_1.parameters(),0.5)
					self.critic_optimizer_1.step()
				elif i == 2:
					self.critic_optimizer_2.zero_grad()
					value_loss_.backward(retain_graph=False)
					grad_norm_value_ = torch.nn.utils.clip_grad_norm_(self.critic_network_2.parameters(),0.5)
					self.critic_optimizer_2.step()
				elif i == 3:
					self.critic_optimizer_3.zero_grad()
					value_loss_.backward(retain_graph=False)
					grad_norm_value_ = torch.nn.utils.clip_grad_norm_(self.critic_network_3.parameters(),0.5)
					self.critic_optimizer_3.step()
				elif i == 4:
					self.critic_optimizer_4.zero_grad()
					value_loss_.backward(retain_graph=False)
					grad_norm_value_ = torch.nn.utils.clip_grad_norm_(self.critic_network_4.parameters(),0.5)
					self.critic_optimizer_4.step()
				elif i == 5:
					self.critic_optimizer_5.zero_grad()
					value_loss_.backward(retain_graph=False)
					grad_norm_value_ = torch.nn.utils.clip_grad_norm_(self.critic_network_5.parameters(),0.5)
					self.critic_optimizer_5.step()
				elif i == 6:
					self.critic_optimizer_6.zero_grad()
					value_loss_.backward(retain_graph=False)
					grad_norm_value_ = torch.nn.utils.clip_grad_norm_(self.critic_network_6.parameters(),0.5)
					self.critic_optimizer_6.step()
				elif i == 7:
					self.critic_optimizer_7.zero_grad()
					value_loss_.backward(retain_graph=False)
					grad_norm_value_ = torch.nn.utils.clip_grad_norm_(self.critic_network_7.parameters(),0.5)
					self.critic_optimizer_7.step()
				elif i == 8:
					self.critic_optimizer_8.zero_grad()
					value_loss_.backward(retain_graph=False)
					grad_norm_value_ = torch.nn.utils.clip_grad_norm_(self.critic_network_8.parameters(),0.5)
					self.critic_optimizer_8.step()
					value_loss.append(None)
					grad_norm_value.append(None)
					weights.append(_weights_)

				value_loss.append(value_loss_)
				grad_norm_value.append(grad_norm_value_)
				weights.append(weights_)

			# # ***********************************************************************************
			# update actor (policy net)
			# # ***********************************************************************************
			entropy = -torch.mean(torch.sum(probs * torch.log(torch.clamp(probs, 1e-10,1.0)), dim=2))

			if self.episode_counter < self.freeze_policy:

				# summing across each agent j to get the advantage
				# so we sum across the second last dimension which does A[t,j] = sum(V[t,i,j] - discounted_rewards[t,i])
				advantage = None
				# V1
				# V_values, weights_ = self.critic_network_1.forward(states_critic, probs.detach(), one_hot_actions)
				# V6
				V_values, weights_ = self.critic_network_6.forward(states_critic, probs.detach(), one_hot_actions)
				V_values = V_values.reshape(-1,self.num_agents,self.num_agents)
				if self.experiment_type == "without_prd":
					advantage = torch.sum(self.calculate_advantages(discounted_rewards, V_values, rewards, dones),dim=-2)
				elif "top" in self.experiment_type:
					values, indices = torch.topk(weights_,k=self.top_k,dim=-1)
					masking_advantage = torch.transpose(torch.sum(F.one_hot(indices, num_classes=self.num_agents), dim=-2),-1,-2)
					advantage = torch.sum(self.calculate_advantages(discounted_rewards, V_values, rewards, dones) * masking_advantage,dim=-2)
				elif self.experiment_type in "above_threshold":
					masking_advantage = torch.transpose((weights_>self.select_above_threshold).int(),-1,-2)
					advantage = torch.sum(self.calculate_advantages(discounted_rewards, V_values, rewards, dones) * masking_advantage,dim=-2)
				elif self.experiment_type == "with_prd_soft_adv":
					advantage = torch.sum(self.calculate_advantages(discounted_rewards, V_values, rewards, dones) * weights_ ,dim=-2)
				elif self.experiment_type == "greedy_policy":
					advantage = torch.sum(self.calculate_advantages(discounted_rewards, V_values, rewards, dones) * self.greedy_policy ,dim=-2)

			
				probs = Categorical(probs)
				policy_loss = -probs.log_prob(actions) * advantage.detach()
				policy_loss = policy_loss.mean() - self.entropy_pen*entropy

				self.policy_optimizer.zero_grad()
				policy_loss.backward(retain_graph=False)
				grad_norm_policy = torch.nn.utils.clip_grad_norm_(self.policy_network.parameters(),0.5)
				self.policy_optimizer.step()

				self.episode_counter += 1


				return value_loss,policy_loss,entropy,grad_norm_value,grad_norm_policy,weights,weight_policy
			else:
				return value_loss,torch.Tensor([5]),entropy,grad_norm_value,-1,weights,-1

		elif "Dual" in self.critic_type:
			V_values_1, weights_1 = self.critic_network_1.forward(states_critic, probs.detach(), one_hot_actions)
			# next_probs, _ = self.policy_network.forward(next_states_actor)
			# V_values_next, _ = self.critic_network.forward(next_states_critic, next_probs.detach(), one_hot_next_actions)
			V_values_1 = V_values_1.reshape(-1,self.num_agents,self.num_agents)
			# V_values_next = V_values_next.reshape(-1,self.num_agents,self.num_agents)


			V_values_2, weights_2 = self.critic_network_2.forward(states_critic, probs.detach(), one_hot_actions)
			# next_probs, _ = self.policy_network.forward(next_states_actor)
			# V_values_next, _ = self.critic_network.forward(next_states_critic, next_probs.detach(), one_hot_next_actions)
			V_values_2 = V_values_2.reshape(-1,self.num_agents,self.num_agents)
			# V_values_next = V_values_next.reshape(-1,self.num_agents,self.num_agents)

		
			# # ***********************************************************************************
			# update critic (value_net)
			# we need a TxNxN vector so inflate the discounted rewards by N --> cloning the discounted rewards for an agent N times
			discounted_rewards = self.calculate_returns(rewards,self.gamma).unsqueeze(-2).repeat(1,self.num_agents,1).to(self.device)
			discounted_rewards = torch.transpose(discounted_rewards,-1,-2)

			# BOOTSTRAP LOSS
			# target_values = torch.transpose(rewards.unsqueeze(-2).repeat(1,self.num_agents,1),-1,-2) + self.gamma*V_values_next*(1-dones.unsqueeze(-1))
			# value_loss = F.smooth_l1_loss(V_values,target_values)

			# MONTE CARLO LOSS
			# value_loss = F.smooth_l1_loss(V_values,discounted_rewards)

			# TD lambda 
			Value_target_1 = self.nstep_returns(V_values_1, rewards, dones).detach()
			value_loss_1 = F.smooth_l1_loss(V_values_1, Value_target_1)

			Value_target_2 = self.nstep_returns(V_values_2, rewards, dones).detach()
			value_loss_2 = F.smooth_l1_loss(V_values_2, Value_target_2)
		
			# # ***********************************************************************************
			# update actor (policy net)
			# # ***********************************************************************************
			entropy = -torch.mean(torch.sum(probs * torch.log(torch.clamp(probs, 1e-10,1.0)), dim=2))

			# summing across each agent j to get the advantage
			# so we sum across the second last dimension which does A[t,j] = sum(V[t,i,j] - discounted_rewards[t,i])
			advantage = None
			if self.experiment_type == "without_prd":
				advantage = torch.sum(self.calculate_advantages(discounted_rewards, Value_target_1, rewards, dones),dim=-2)
			elif "top" in self.experiment_type:
				values, indices = torch.topk(weights_2,k=self.top_k,dim=-1)
				masking_advantage = torch.transpose(torch.sum(F.one_hot(indices, num_classes=self.num_agents), dim=-2),-1,-2)
				advantage = torch.sum(self.calculate_advantages(discounted_rewards, V_values_1, rewards, dones) * masking_advantage,dim=-2)
			elif self.experiment_type in "above_threshold":
				masking_advantage = torch.transpose((weights_2>self.select_above_threshold).int(),-1,-2)
				advantage = torch.sum(self.calculate_advantages(discounted_rewards, V_values_1, rewards, dones) * masking_advantage,dim=-2)
			elif self.experiment_type == "with_prd_soft_adv":
				advantage = torch.sum(self.calculate_advantages(discounted_rewards, V_values_1, rewards, dones) * weights_2 ,dim=-2)
			elif self.experiment_type == "greedy_policy":
				advantage = torch.sum(self.calculate_advantages(discounted_rewards, V_values_1, rewards, dones) * self.greedy_policy ,dim=-2)

		
			probs = Categorical(probs)
			policy_loss = -probs.log_prob(actions) * advantage.detach()
			policy_loss = policy_loss.mean() - self.entropy_pen*entropy
			# # ***********************************************************************************
				
			# **********************************
			self.critic_optimizer_1.zero_grad()
			value_loss_1.backward(retain_graph=False)
			grad_norm_value_1 = torch.nn.utils.clip_grad_norm_(self.critic_network_1.parameters(),0.5)
			self.critic_optimizer_1.step()

			self.critic_optimizer_2.zero_grad()
			value_loss_2.backward(retain_graph=False)
			grad_norm_value_2 = torch.nn.utils.clip_grad_norm_(self.critic_network_2.parameters(),0.5)
			self.critic_optimizer_2.step()


			self.policy_optimizer.zero_grad()
			policy_loss.backward(retain_graph=False)
			grad_norm_policy = torch.nn.utils.clip_grad_norm_(self.policy_network.parameters(),0.5)
			self.policy_optimizer.step()

			# V values
			return [value_loss_1, value_loss_2],policy_loss,entropy,[grad_norm_value_1, grad_norm_value_2],grad_norm_policy,[weights_1, weights_2],weight_policy

		else:
			V_values, weights = self.critic_network.forward(states_critic, probs.detach(), one_hot_actions)
			# next_probs, _ = self.policy_network.forward(next_states_actor)
			# V_values_next, _ = self.critic_network.forward(next_states_critic, next_probs.detach(), one_hot_next_actions)
			V_values = V_values.reshape(-1,self.num_agents,self.num_agents)
			# V_values_next = V_values_next.reshape(-1,self.num_agents,self.num_agents)

		
			# # ***********************************************************************************
			# update critic (value_net)
			# we need a TxNxN vector so inflate the discounted rewards by N --> cloning the discounted rewards for an agent N times
			discounted_rewards = self.calculate_returns(rewards,self.gamma).unsqueeze(-2).repeat(1,self.num_agents,1).to(self.device)
			discounted_rewards = torch.transpose(discounted_rewards,-1,-2)

			# BOOTSTRAP LOSS
			# target_values = torch.transpose(rewards.unsqueeze(-2).repeat(1,self.num_agents,1),-1,-2) + self.gamma*V_values_next*(1-dones.unsqueeze(-1))
			# value_loss = F.smooth_l1_loss(V_values,target_values)

			# MONTE CARLO LOSS
			# value_loss = F.smooth_l1_loss(V_values,discounted_rewards)

			# TD lambda 
			Value_target = self.nstep_returns(V_values, rewards, dones).detach()
			value_loss = F.smooth_l1_loss(V_values, Value_target)

			weights_off_diagonal = weights * (1 - torch.eye(self.num_agents,device=self.device))
			l1_weights = torch.mean(weights_off_diagonal)
			value_loss += self.l1_pen*l1_weights
		
			# # ***********************************************************************************
			# update actor (policy net)
			# # ***********************************************************************************
			entropy = -torch.mean(torch.sum(probs * torch.log(torch.clamp(probs, 1e-10,1.0)), dim=2))

			# summing across each agent j to get the advantage
			# so we sum across the second last dimension which does A[t,j] = sum(V[t,i,j] - discounted_rewards[t,i])
			advantage = None
			if self.experiment_type == "without_prd":
				advantage = torch.sum(self.calculate_advantages(discounted_rewards, V_values, rewards, dones),dim=-2)
			elif "top" in self.experiment_type:
				values, indices = torch.topk(weights,k=self.top_k,dim=-1)
				masking_advantage = torch.sum(F.one_hot(indices, num_classes=self.num_agents), dim=-2)
				advantage = torch.sum(self.calculate_advantages(discounted_rewards, V_values, rewards, dones) * masking_advantage,dim=-2)
			elif "above_threshold" in self.experiment_type:
				masking_advantage = (weights>self.select_above_threshold).int()
				advantage = torch.sum(self.calculate_advantages(discounted_rewards, V_values, rewards, dones) * masking_advantage,dim=-2)
			elif "with_prd_soft_adv" in self.experiment_type:
				advantage = torch.sum(self.calculate_advantages(discounted_rewards, V_values, rewards, dones) * weights ,dim=-2)
			elif "with_prd_averaged" in self.experiment_type:
				avg_weights = torch.mean(weights,dim=0)
				advantage = torch.sum(self.calculate_advantages(discounted_rewards, V_values, rewards, dones) * avg_weights ,dim=-2)
			elif self.experiment_type == "greedy_policy":
				advantage = torch.sum(self.calculate_advantages(discounted_rewards, V_values, rewards, dones) * self.greedy_policy ,dim=-2)

			if "scaled" in self.experiment_type:
				if "with_prd_soft_adv" in self.experiment_type:
					advantage = advantage*self.num_agents
				elif "top" in self.experiment_type:
					advantage = advantage*(self.num_agents/self.top_k)
		
			probs = Categorical(probs)
			policy_loss = -probs.log_prob(actions) * advantage.detach()
			policy_loss = policy_loss.mean() - self.entropy_pen*entropy
			# # ***********************************************************************************
				
			# **********************************
			self.critic_optimizer.zero_grad()
			value_loss.backward(retain_graph=False)
			if "AttentionCritic" in self.critic_type:
				self.critic_network.scale_shared_grads()
			# SCALE GRADS
			# self.critic_network.scale_shared_grads()
			grad_norm_value = torch.nn.utils.clip_grad_norm_(self.critic_network.parameters(),0.5)
			self.critic_optimizer.step()


			self.policy_optimizer.zero_grad()
			policy_loss.backward(retain_graph=False)
			grad_norm_policy = torch.nn.utils.clip_grad_norm_(self.policy_network.parameters(),0.5)
			self.policy_optimizer.step()

			# V values
			return value_loss,policy_loss,entropy,grad_norm_value,grad_norm_policy,weights,weight_policy