import numpy as np
import torch
import torch.optim as optim
from torch.distributions import Categorical
from ppo_model import *
import torch.nn.functional as F


class PPOAgent:
	def __init__(self, 
		env,
		dictionary):

		self.env = env
		self.env_name = dictionary["env"]
		self.value_lr = dictionary["value_lr"]
		self.policy_lr = dictionary["policy_lr"]
		self.gamma = dictionary["gamma"]
		self.entropy_pen = dictionary["entropy_pen"]
		self.trace_decay = dictionary["trace_decay"]
		self.top_k = dictionary["top_k"]
		self.gae = dictionary["gae"]
		self.critic_loss_type = dictionary["critic_loss_type"]
		self.norm_adv = dictionary["norm_adv"]
		self.norm_rew = dictionary["norm_rew"]
		# Used for masking advantages above a threshold
		self.select_above_threshold = dictionary["select_above_threshold"]
		self.policy_clip = dictionary["policy_clip"]
		self.n_epochs = dictionary["n_epochs"]
		

		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		# self.device = "cpu"
		
		self.num_agents = self.env.n
		self.num_actions = self.env.action_space[0].n
		self.gif = dictionary["gif"]

		self.experiment_type = dictionary["experiment_type"]

		self.greedy_policy = torch.zeros(self.num_agents,self.num_agents).to(self.device)
		for i in range(self.num_agents):
			self.greedy_policy[i][i] = 1

		print("EXPERIMENT TYPE", self.experiment_type)

		# TD lambda
		self.lambda_ = dictionary["lambda"]

		# PAIRED AGENT
		if self.env_name == "paired_by_sharing_goals":
			obs_dim = 2*4
		elif self.env_name == "crossing":
			obs_dim = 2*3
		elif self.env_name in ["color_social_dilemma", "color_social_dilemma_pt2"]:
			obs_dim = 2*2 + 1 + 2*3

		self.buffer = RolloutBuffer()

		self.critic_network = GATCritic(obs_dim, 128, obs_dim+self.num_actions, 128, 128, 1, self.num_agents, self.num_actions).to(self.device)
		
		
		if self.env_name in ["paired_by_sharing_goals", "crossing"]:
			obs_dim = 2*3
		elif self.env_name in ["color_social_dilemma", "color_social_dilemma_pt2"]:
			obs_dim = 2*2 + 1 + 2*3

		# MLP POLICY
		self.policy_network = MLPPolicyNetwork(obs_dim, self.num_agents, self.num_actions).to(self.device)
		self.policy_network_old = MLPPolicyNetwork(obs_dim, self.num_agents, self.num_actions).to(self.device)

		# COPY
		self.policy_network_old.load_state_dict(self.policy_network.state_dict())


		# Loading models
		# model_path_value = "../../../tests/color_social_dilemma/models/color_social_dilemma_with_prd_above_threshold_0.01_MLPToGNNV6_color_social_dilemma_try2/critic_networks/20-07-2021VN_ATN_FCN_lr0.001_PN_ATN_FCN_lr0.0005_GradNorm0.5_Entropy0.008_trace_decay0.98topK_0select_above_threshold0.01softmax_cut_threshold0.1_epsiode200000_MLPToGNNV6.pt"
		# model_path_policy = "../../../tests/color_social_dilemma/models/color_social_dilemma_with_prd_above_threshold_0.01_MLPToGNNV6_color_social_dilemma_try2/actor_networks/20-07-2021_PN_ATN_FCN_lr0.0005VN_SAT_FCN_lr0.001_GradNorm0.5_Entropy0.008_trace_decay0.98topK_0select_above_threshold0.01softmax_cut_threshold0.1_epsiode200000_MLPToGNNV6.pt"
		# For CPU
		# self.critic_network.load_state_dict(torch.load(model_path_value,map_location=torch.device('cpu')))
		# self.policy_network.load_state_dict(torch.load(model_path_policy,map_location=torch.device('cpu')))
		# # For GPU
		# self.critic_network.load_state_dict(torch.load(model_path_value))
		# self.policy_network.load_state_dict(torch.load(model_path_policy))

		
		self.critic_optimizer = optim.Adam(self.critic_network.parameters(),lr=self.value_lr)
		self.policy_optimizer = optim.Adam(self.policy_network.parameters(),lr=self.policy_lr)



	def calculate_advantages(self,returns, values_old, values_new, rewards, dones):

		advantages = None

		if self.gae:
			advantages = []
			next_value = 0
			advantage = 0
			rewards = rewards.unsqueeze(-1)
			dones = dones.unsqueeze(-1)
			masks = 1 - dones

			for t in reversed(range(0, len(rewards))):
				td_error = rewards[t] + (self.gamma * next_value * masks[t]) - values_new.data[t]
				# print("td_error")
				# print(td_error.shape)
				next_value = values_old.data[t]
				
				advantage = td_error + (self.gamma * self.trace_decay * advantage * masks[t])
				# print("advantage")
				# print(advantage.shape)
				advantages.insert(0, advantage)

			advantages = torch.stack(advantages)	
		else:
			advantages = returns - values_new
		
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


	def calculate_returns(self, rewards, dones):

		R = 0
		returns = []

		for reward, done in zip(reversed(rewards), reversed(dones)):
			if all(done):
				R = 0
			R = reward + (self.gamma * R)
			returns.insert(0, R)

		returns_tensor = torch.stack(returns).to(self.device)
		
		if self.norm_rew:
			
			returns_tensor = (returns_tensor - returns_tensor.mean()) / returns_tensor.std()
			
		return returns_tensor


	def get_action(self,state):

		with torch.no_grad():
			state = torch.FloatTensor([state]).to(self.device)
			dists, _ = self.policy_network_old(state)
			actions = [Categorical(dist).sample().detach().cpu().item() for dist in dists[0]]

			probs = Categorical(dists)
			action_logprob = probs.log_prob(torch.FloatTensor(actions).to(self.device))

			self.buffer.probs.append(dists.detach().cpu())
			self.buffer.logprobs.append(action_logprob.detach().cpu())

			return actions



	def update(self):

		# convert list to tensor
		old_states_critic = torch.FloatTensor(self.buffer.states_critic).to(self.device)
		old_states_actor = torch.FloatTensor(self.buffer.states_actor).to(self.device)
		old_actions = torch.FloatTensor(self.buffer.actions).to(self.device)
		old_one_hot_actions = torch.FloatTensor(self.buffer.one_hot_actions).to(self.device)
		old_probs = torch.stack(self.buffer.probs).squeeze(1).to(self.device)
		old_logprobs = torch.stack(self.buffer.logprobs).squeeze(1).to(self.device)
		rewards = torch.FloatTensor(self.buffer.rewards).to(self.device)
		dones = torch.FloatTensor(self.buffer.dones).to(self.device)

		V_values_old, weights = self.critic_network(old_states_critic, old_probs, old_one_hot_actions)
		V_values_old = V_values_old.reshape(-1,self.num_agents,self.num_agents)
		
		if self.critic_loss_type == "MC":
			discounted_rewards = self.calculate_returns(rewards, dones).unsqueeze(-2).repeat(1,self.num_agents,1).to(self.device)
			Value_target = torch.transpose(discounted_rewards,-1,-2)
		elif self.critic_loss_type == "TD_lambda":
			Value_target = self.nstep_returns(V_values_old, rewards, dones).detach()

		value_loss_batch = 0
		policy_loss_batch = 0
		entropy_batch = 0
		value_weights_batch = torch.zeros_like(weights)
		policy_weights_batch = torch.zeros_like(weights)
		grad_norm_value_batch = 0
		grad_norm_policy_batch = 0
		
		# Optimize policy for n epochs
		for _ in range(self.n_epochs):

			V_values, weights = self.critic_network(old_states_critic, old_probs, old_one_hot_actions)
			V_values = V_values.reshape(-1,self.num_agents,self.num_agents)


			# summing across each agent j to get the advantage
			# so we sum across the second last dimension which does A[t,j] = sum(V[t,i,j] - discounted_rewards[t,i])
			advantage = None
			if self.experiment_type == "shared":
				advantage = torch.sum(self.calculate_advantages(Value_target, V_values_old, V_values, rewards, dones),dim=-2)
			elif "prd_soft_adv" in self.experiment_type:
				advantage = torch.sum(self.calculate_advantages(Value_target, V_values_old, V_values, rewards, dones) * weights ,dim=-2)
			elif "prd_averaged" in self.experiment_type:
				avg_weights = torch.mean(weights,dim=0)
				advantage = torch.sum(self.calculate_advantages(Value_target, V_values_old, V_values, rewards, dones) * avg_weights ,dim=-2)
			elif "prd_avg_top" in self.experiment_type:
				avg_weights = torch.mean(weights,dim=0)
				values, indices = torch.topk(avg_weights,k=self.top_k,dim=-1)
				masking_advantage = torch.sum(F.one_hot(indices, num_classes=self.num_agents), dim=-2)
				advantage = torch.sum(self.calculate_advantages(Value_target, V_values_old, V_values, rewards, dones) * masking_advantage,dim=-2)
			elif "prd_avg_above_threshold" in self.experiment_type:
				avg_weights = torch.mean(weights,dim=0)
				masking_advantage = (avg_weights>self.select_above_threshold).int()
				advantage = torch.sum(self.calculate_advantages(Value_target, V_values_old, V_values, rewards, dones) * masking_advantage,dim=-2)
			elif "above_threshold" in self.experiment_type:
				masking_advantage = (weights>self.select_above_threshold).int()
				advantage = torch.sum(self.calculate_advantages(Value_target, V_values_old, V_values, rewards, dones) * masking_advantage,dim=-2)
			elif "top" in self.experiment_type:
				values, indices = torch.topk(weights,k=self.top_k,dim=-1)
				masking_advantage = torch.sum(F.one_hot(indices, num_classes=self.num_agents), dim=-2)
				advantage = torch.sum(self.calculate_advantages(Value_target, V_values_old, V_values, rewards, dones) * masking_advantage,dim=-2)
			elif self.experiment_type == "greedy":
				advantage = torch.sum(self.calculate_advantages(Value_target, V_values_old, V_values, rewards, dones) * self.greedy_policy ,dim=-2)


			# Evaluating old actions and values
			dists, weights_policy = self.policy_network(old_states_actor)
			probs = Categorical(dists)
			logprobs = probs.log_prob(old_actions)

			# Finding the ratio (pi_theta / pi_theta__old)
			ratios = torch.exp(logprobs - old_logprobs)
			# Finding Surrogate Loss
			surr1 = ratios * advantage.detach()
			surr2 = torch.clamp(ratios, 1-self.policy_clip, 1+self.policy_clip) * advantage.detach()

			# final loss of clipped objective PPO
			entropy = -torch.mean(torch.sum(dists * torch.log(torch.clamp(dists, 1e-10,1.0)), dim=2))
			policy_loss = -torch.min(surr1, surr2).mean() - self.entropy_pen*entropy


			critic_loss = F.smooth_l1_loss(V_values, Value_target)
			
			
			# take gradient step
			self.critic_optimizer.zero_grad()
			critic_loss.backward()
			grad_norm_value = torch.nn.utils.clip_grad_norm_(self.critic_network.parameters(),0.5)
			self.critic_optimizer.step()

			self.policy_optimizer.zero_grad()
			policy_loss.backward()
			grad_norm_policy = torch.nn.utils.clip_grad_norm_(self.policy_network.parameters(),0.5)
			self.policy_optimizer.step()


			value_loss_batch += critic_loss.item()
			policy_loss_batch += policy_loss.item()
			entropy_batch += entropy.item()
			grad_norm_value_batch += grad_norm_value
			grad_norm_policy_batch += grad_norm_policy
			value_weights_batch += weights.detach()
			policy_weights_batch += weights_policy.detach()
			
		# Copy new weights into old policy
		self.policy_network_old.load_state_dict(self.policy_network.state_dict())

		# clear buffer
		self.buffer.clear()

		value_loss_batch /= self.n_epochs
		policy_loss_batch /= self.n_epochs
		entropy_batch /= self.n_epochs
		grad_norm_value_batch /= self.n_epochs
		grad_norm_policy_batch /= self.n_epochs
		value_weights_batch /= self.n_epochs
		policy_weights_batch /= self.n_epochs
		

		return value_loss_batch,policy_loss_batch,entropy_batch,grad_norm_value_batch,grad_norm_policy_batch,value_weights_batch,policy_weights_batch
		
	   
