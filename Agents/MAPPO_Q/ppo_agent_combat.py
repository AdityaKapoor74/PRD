import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
from torch.distributions import Categorical
from ppo_model_combat import *
import torch.nn.functional as F

class PPOAgent_COMBAT:

	def __init__(
		self, 
		env, 
		dictionary,
		comet_ml,
		):

		self.env = env
		self.update_learning_rate_with_prd = dictionary["update_learning_rate_with_prd"]
		self.test_num = dictionary["test_num"]
		self.env_name = dictionary["env"]
		self.experiment_type = dictionary["experiment_type"]
		self.gif = dictionary["gif"]
		self.prd_num_agents = self.env.n_agents
		self.shared_num_agents = self.env._n_opponents
		self.num_actions = self.env.action_space[0].n
		self.n_epochs = dictionary["n_epochs"]
		self.value_normalization = dictionary["value_normalization"]
		self.norm_adv = dictionary["norm_adv"]
		self.norm_returns = dictionary["norm_returns"]
		self.gae_lambda = dictionary["gae_lambda"]
		self.gamma = dictionary["gamma"]
		self.lambda_ = dictionary["lambda_"]
		# SHARED
		self.shared_value_lr = dictionary["shared_value_lr"]
		self.shared_policy_lr = dictionary["shared_policy_lr"]
		self.shared_entropy_pen = dictionary["shared_entropy_pen"]
		self.shared_critic_weight_entropy_pen = dictionary["shared_critic_weight_entropy_pen"]
		# TD lambda
		self.shared_policy_clip = dictionary["shared_policy_clip"]
		self.shared_value_clip = dictionary["shared_value_clip"]
		self.shared_grad_clip_critic = dictionary["shared_grad_clip_critic"]
		self.shared_grad_clip_actor = dictionary["shared_grad_clip_actor"]

		# episode track
		self.episode = 0

		# PRD
		self.prd_type = dictionary["prd_type"]
		self.prd_value_lr = dictionary["prd_value_lr"]
		self.prd_policy_lr = dictionary["prd_policy_lr"]
		self.prd_entropy_pen = dictionary["prd_entropy_pen"]
		self.prd_critic_weight_entropy_pen = dictionary["prd_critic_weight_entropy_pen"]
		# TD lambda
		self.prd_policy_clip = dictionary["prd_policy_clip"]
		self.prd_value_clip = dictionary["prd_value_clip"]
		self.prd_grad_clip_critic = dictionary["prd_grad_clip_critic"]
		self.prd_grad_clip_actor = dictionary["prd_grad_clip_actor"]
		# Used for masking advantages above a threshold
		self.top_k = dictionary["top_k"]
		self.select_above_threshold = dictionary["select_above_threshold"]
		self.threshold_min = dictionary["threshold_min"]
		self.threshold_max = dictionary["threshold_max"]
		self.steps_to_take = dictionary["steps_to_take"]


		if "prd_above_threshold_decay" in self.prd_type:
			self.threshold_delta = (self.select_above_threshold - self.threshold_min)/self.steps_to_take
		elif "prd_above_threshold_ascend" in self.prd_type:
			self.threshold_delta = (self.threshold_max - self.select_above_threshold)/self.steps_to_take


		if dictionary["device"] == "gpu":
			self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		else:
			self.device = "cpu"

		self.seeds = [42, 142, 242, 342, 442]
		torch.manual_seed(self.seeds[dictionary["iteration"]-1])

		# SHARED AGENT
		print("EXPERIMENT TYPE", self.experiment_type)
		self.shared_critic_network = Q_network(in_channels = 6, num_agents=self.shared_num_agents, num_actions=self.num_actions, value_normalization=self.value_normalization, device=self.device).to(self.device)
		self.shared_critic_network_old = Q_network(in_channels = 6, num_agents=self.shared_num_agents, num_actions=self.num_actions, value_normalization=self.value_normalization, device=self.device).to(self.device)
		for param in self.shared_critic_network_old.parameters():
			param.requires_grad_(False)
		# COPY
		self.shared_critic_network_old.load_state_dict(self.shared_critic_network.state_dict())
		
		# POLICY
		self.shared_policy_network = Policy(in_channels = 6, num_agents=self.shared_num_agents, num_actions=self.num_actions, device=self.device).to(self.device)
		self.shared_policy_network_old = Policy(in_channels = 6, num_agents=self.shared_num_agents, num_actions=self.num_actions, device=self.device).to(self.device)
		for param in self.shared_policy_network_old.parameters():
			param.requires_grad_(False)
		# COPY
		self.shared_policy_network_old.load_state_dict(self.shared_policy_network.state_dict())

		# PRD AGENT
		self.prd_critic_network = Q_network(in_channels = 6, num_agents=self.prd_num_agents, num_actions=self.num_actions, value_normalization=self.value_normalization, device=self.device).to(self.device)
		self.prd_critic_network_old = Q_network(in_channels = 6, num_agents=self.prd_num_agents, num_actions=self.num_actions, value_normalization=self.value_normalization, device=self.device).to(self.device)
		for param in self.prd_critic_network_old.parameters():
			param.requires_grad_(False)
		# COPY
		self.prd_critic_network_old.load_state_dict(self.prd_critic_network.state_dict())
		
		# POLICY
		self.prd_policy_network = Policy(in_channels = 6, num_agents=self.prd_num_agents, num_actions=self.num_actions, device=self.device).to(self.device)
		self.prd_policy_network_old = Policy(in_channels = 6, num_agents=self.prd_num_agents, num_actions=self.num_actions, device=self.device).to(self.device)
		for param in self.prd_policy_network_old.parameters():
			param.requires_grad_(False)
		# COPY
		self.prd_policy_network_old.load_state_dict(self.prd_policy_network.state_dict())


		self.buffer = RolloutBuffer()


		if dictionary["load_models"]:
			# Loading models
			self.shared_critic_network.load_state_dict(torch.load(dictionary["shared_model_path_value"],map_location=self.device))
			self.shared_policy_network.load_state_dict(torch.load(dictionary["shared_model_path_policy"],map_location=self.device))
			self.prd_critic_network.load_state_dict(torch.load(dictionary["prd_model_path_value"],map_location=self.device))
			self.prd_policy_network.load_state_dict(torch.load(dictionary["prd_model_path_policy"],map_location=self.device))

		
		self.shared_critic_optimizer = optim.Adam(self.shared_critic_network.parameters(),lr=self.shared_value_lr)
		self.shared_policy_optimizer = optim.Adam(self.shared_policy_network.parameters(),lr=self.shared_policy_lr)
		self.prd_critic_optimizer = optim.Adam(self.prd_critic_network.parameters(),lr=self.prd_value_lr)
		self.prd_policy_optimizer = optim.Adam(self.prd_policy_network.parameters(),lr=self.prd_policy_lr)


		# self.scheduler = optim.lr_scheduler.MultiStepLR(self.policy_optimizer, milestones=[1000], gamma=2)


		self.comet_ml = None
		if dictionary["save_comet_ml_plot"]:
			self.comet_ml = comet_ml


	def get_action(self, prd_states, shared_states, greedy=False):
		with torch.no_grad():
			shared_states = torch.from_numpy(shared_states).float().to(self.device).unsqueeze(0)
			dists = self.shared_policy_network_old(shared_states).squeeze(0)
			if greedy:
				shared_actions = [dist.argmax().detach().cpu().item() for dist in dists]
			else:
				shared_actions = [Categorical(dist).sample().detach().cpu().item() for dist in dists]

			probs = Categorical(dists)
			action_logprob = probs.log_prob(torch.FloatTensor(shared_actions).to(self.device))

			self.buffer.shared_probs.append(dists.detach().cpu())
			self.buffer.shared_logprobs.append(action_logprob.detach().cpu())

			prd_states = torch.from_numpy(prd_states).float().to(self.device).unsqueeze(0)
			dists = self.prd_policy_network_old(prd_states).squeeze(0)
			if greedy:
				prd_actions = [dist.argmax().detach().cpu().item() for dist in dists]
			else:
				prd_actions = [Categorical(dist).sample().detach().cpu().item() for dist in dists]

			probs = Categorical(dists)
			action_logprob = probs.log_prob(torch.FloatTensor(prd_actions).to(self.device))

			self.buffer.prd_probs.append(dists.detach().cpu())
			self.buffer.prd_logprobs.append(action_logprob.detach().cpu())

			return prd_actions, shared_actions


	def calculate_advantages(self, values, rewards, dones):
		advantages = []
		next_value = 0
		advantage = 0
		rewards = rewards.unsqueeze(-1)
		dones = dones.unsqueeze(-1)
		masks = 1 - dones
		for t in reversed(range(0, len(rewards))):
			td_error = rewards[t] + (self.gamma * next_value * masks[t]) - values.data[t]
			next_value = values.data[t]
			
			advantage = td_error + (self.gamma * self.gae_lambda * advantage * masks[t])
			advantages.insert(0, advantage)

		advantages = torch.stack(advantages)
		
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
		
		if self.norm_returns:
			returns_tensor = (returns_tensor - returns_tensor.mean()) / returns_tensor.std()
			
		return returns_tensor


	def plot(self, episode):

		self.comet_ml.log_metric('PRD Value_Loss',self.plotting_dict["prd_value_loss"].item(),episode)
		self.comet_ml.log_metric('Shared Value_Loss',self.plotting_dict["shared_value_loss"].item(),episode)
		self.comet_ml.log_metric('PRD Grad_Norm_Value',self.plotting_dict["prd_grad_norm_value"],episode)
		self.comet_ml.log_metric('Shared Grad_Norm_Value',self.plotting_dict["shared_grad_norm_value"],episode)
		self.comet_ml.log_metric('PRD Policy_Loss',self.plotting_dict["prd_policy_loss"].item(),episode)
		self.comet_ml.log_metric('Shared Policy_Loss',self.plotting_dict["shared_policy_loss"].item(),episode)
		self.comet_ml.log_metric('PRD Grad_Norm_Policy',self.plotting_dict["prd_grad_norm_policy"],episode)
		self.comet_ml.log_metric('Shared Grad_Norm_Policy',self.plotting_dict["shared_grad_norm_policy"],episode)
		self.comet_ml.log_metric('PRD Entropy',self.plotting_dict["prd_entropy"].item(),episode)
		self.comet_ml.log_metric('Shared Entropy',self.plotting_dict["shared_entropy"].item(),episode)

		if "threshold" in self.prd_type:
			for i in range(self.prd_num_agents):
				agent_name = "agent"+str(i)
				self.comet_ml.log_metric('Group_Size_'+agent_name, self.plotting_dict["prd_agent_groups_over_episode"][i].item(), episode)

			self.comet_ml.log_metric('Avg_Group_Size', self.plotting_dict["prd_avg_agent_group_over_episode"].item(), episode)

		if "prd_top" in self.prd_type:
			self.comet_ml.log_metric('Mean_Smallest_Weight', self.plotting_dict["prd_mean_min_weight_value"].item(), episode)


		# ENTROPY OF WEIGHTS
		prd_entropy_weights = -torch.mean(torch.sum(self.plotting_dict["prd_weights_value"]* torch.log(torch.clamp(self.plotting_dict["prd_weights_value"], 1e-10,1.0)), dim=2))
		self.comet_ml.log_metric('PRD Critic_Weight_Entropy', prd_entropy_weights.item(), episode)
		shared_entropy_weights = -torch.mean(torch.sum(self.plotting_dict["shared_weights_value"]* torch.log(torch.clamp(self.plotting_dict["shared_weights_value"], 1e-10,1.0)), dim=2))
		self.comet_ml.log_metric('Shared Critic_Weight_Entropy', shared_entropy_weights.item(), episode)
		

	def calculate_advantages_based_on_exp(self, V_values, rewards, dones, weights_prd, episode):
		advantage = None
		masking_advantage = None
		mean_min_weight_value = -1
		if "shared" in self.prd_type:
			advantage = torch.sum(self.calculate_advantages(V_values, rewards, dones),dim=-2)
		elif "prd_above_threshold" in self.prd_type:
			masking_advantage = (weights_prd>self.select_above_threshold).int()
			advantage = torch.sum(self.calculate_advantages(V_values, rewards, dones) * torch.transpose(masking_advantage,-1,-2),dim=-2)
		elif "top" in self.prd_type:
			if episode < self.steps_to_take:
				advantage = torch.sum(self.calculate_advantages(V_values, rewards, dones),dim=-2)
				min_weight_values, _ = torch.min(weights_prd, dim=-1)
				mean_min_weight_value = torch.mean(min_weight_values)
			else:
				values, indices = torch.topk(weights_prd,k=self.top_k,dim=-1)
				min_weight_values, _ = torch.min(values, dim=-1)
				mean_min_weight_value = torch.mean(min_weight_values)
				masking_advantage = torch.sum(F.one_hot(indices, num_classes=self.num_agents), dim=-2)
				advantage = torch.sum(self.calculate_advantages(V_values, rewards, dones) * torch.transpose(masking_advantage,-1,-2),dim=-2)
		elif "greedy" in self.prd_type:
			advantage = torch.sum(self.calculate_advantages(V_values, rewards, dones) * self.greedy_policy ,dim=-2)

		if "scaled" in self.prd_type and episode > self.steps_to_take and "top" in self.prd_type:
			advantage = advantage*(self.num_agents/self.top_k)

		return advantage, masking_advantage, mean_min_weight_value

	def update_parameters(self):
		# increment episode
		self.episode+=1

		if self.select_above_threshold > self.threshold_min and "prd_above_threshold_decay" in self.prd_type:
			self.select_above_threshold = self.select_above_threshold - self.threshold_delta

		if self.threshold_max >= self.select_above_threshold and "prd_above_threshold_ascend" in self.prd_type:
			self.select_above_threshold = self.select_above_threshold + self.threshold_delta



	def update(self,episode):
		# PRD
		prd_old_states = torch.FloatTensor(np.array(self.buffer.prd_states)).to(self.device)
		prd_old_actions = torch.FloatTensor(np.array(self.buffer.prd_actions)).to(self.device)
		prd_old_one_hot_actions = torch.FloatTensor(np.array(self.buffer.prd_one_hot_actions)).to(self.device)
		prd_old_probs = torch.stack(self.buffer.prd_probs, dim=0).to(self.device)
		prd_old_logprobs = torch.stack(self.buffer.prd_logprobs, dim=0).to(self.device)
		prd_rewards = torch.FloatTensor(np.array(self.buffer.prd_rewards)).to(self.device)
		prd_dones = torch.FloatTensor(np.array(self.buffer.prd_dones)).long().to(self.device)
		# SHARED
		shared_old_states = torch.FloatTensor(np.array(self.buffer.shared_states)).to(self.device)
		shared_old_actions = torch.FloatTensor(np.array(self.buffer.shared_actions)).to(self.device)
		shared_old_one_hot_actions = torch.FloatTensor(np.array(self.buffer.shared_one_hot_actions)).to(self.device)
		shared_old_probs = torch.stack(self.buffer.shared_probs, dim=0).to(self.device)
		shared_old_logprobs = torch.stack(self.buffer.shared_logprobs, dim=0).to(self.device)
		shared_rewards = torch.FloatTensor(np.array(self.buffer.shared_rewards)).to(self.device)
		shared_dones = torch.FloatTensor(np.array(self.buffer.shared_dones)).long().to(self.device)


		prd_Values_old, prd_Q_values_old, prd_weights_value_old = self.prd_critic_network_old(prd_old_states, prd_old_probs.squeeze(-2), prd_old_one_hot_actions)
		prd_Values_old = prd_Values_old.reshape(-1,self.prd_num_agents,self.prd_num_agents)

		shared_Values_old, shared_Q_values_old, shared_weights_value_old = self.shared_critic_network_old(shared_old_states, shared_old_probs.squeeze(-2), shared_old_one_hot_actions)
		shared_Values_old = shared_Values_old.reshape(-1,self.shared_num_agents,self.shared_num_agents)
		

		if self.value_normalization:
			prd_Q_values_old = torch.sum(self.prd_critic_network_old.pop_art.denormalize(prd_Q_values_old)*prd_old_one_hot_actions, dim=-1).unsqueeze(-1)
			shared_Q_values_old = torch.sum(self.shared_critic_network_old.pop_art.denormalize(shared_Q_values_old)*shared_old_one_hot_actions, dim=-1).unsqueeze(-1)
		
		prd_Q_value_target = self.nstep_returns(prd_Q_values_old, prd_rewards, prd_dones).detach()
		shared_Q_value_target = self.nstep_returns(shared_Q_values_old, shared_rewards, shared_dones).detach()

		prd_value_loss_batch, shared_value_loss_batch = 0, 0
		prd_policy_loss_batch, shared_policy_loss_batch = 0, 0
		prd_entropy_batch, shared_entropy_batch = 0, 0
		prd_value_weights_batch, shared_value_weights_batch = None, None
		prd_grad_norm_value_batch, shared_grad_norm_value_batch = 0, 0
		prd_grad_norm_policy_batch, shared_grad_norm_policy_batch = 0, 0
		prd_agent_groups_over_episode_batch = 0
		prd_avg_agent_group_over_episode_batch = 0

		# torch.autograd.set_detect_anomaly(True)
		# Optimize policy for n epochs
		for _ in range(self.n_epochs):

			prd_Value, prd_Q_value, prd_weights_value = self.prd_critic_network(prd_old_states, prd_old_probs.squeeze(-2), prd_old_one_hot_actions)
			prd_Value = prd_Value.reshape(-1,self.prd_num_agents,self.prd_num_agents)
			shared_Value, shared_Q_value, shared_weights_value = self.shared_critic_network(shared_old_states, shared_old_probs.squeeze(-2), shared_old_one_hot_actions)
			shared_Value = shared_Value.reshape(-1,self.shared_num_agents,self.shared_num_agents)

			prd_advantage, prd_masking_advantage, prd_mean_min_weight_value = self.calculate_advantages_based_on_exp(prd_Value, prd_rewards, prd_dones, prd_weights_value, episode)
			shared_advantage, shared_masking_advantage, shared_mean_min_weight_value = self.calculate_advantages_based_on_exp(shared_Value, shared_rewards, shared_dones, shared_weights_value, episode)

			if "threshold" in self.prd_type:
				agent_groups_over_episode = torch.sum(torch.sum(prd_masking_advantage.float(), dim=-2),dim=0)/prd_masking_advantage.shape[0]
				avg_agent_group_over_episode = torch.mean(agent_groups_over_episode)
				prd_agent_groups_over_episode_batch += agent_groups_over_episode
				prd_avg_agent_group_over_episode_batch += avg_agent_group_over_episode

			prd_dists = self.prd_policy_network(prd_old_states)
			prd_probs = Categorical(prd_dists.squeeze(0))
			prd_logprobs = prd_probs.log_prob(prd_old_actions)

			shared_dists = self.shared_policy_network(shared_old_states)
			shared_probs = Categorical(shared_dists.squeeze(0))
			shared_logprobs = shared_probs.log_prob(shared_old_actions)

			if self.value_normalization:
				self.prd_critic_network.pop_art.update(prd_Q_value_target)
				prd_Q_value_target_normalized = torch.sum(self.prd_critic_network.pop_art.normalize(prd_Q_value_target)*prd_old_one_hot_actions, dim=-1).unsqueeze(-1) # gives for all possible actions
				prd_critic_loss_1 = F.smooth_l1_loss(prd_Q_value,prd_Q_value_target_normalized)
				prd_critic_loss_2 = F.smooth_l1_loss(torch.clamp(prd_Q_value_target_normalized, Q_values_old-self.prd_value_clip, prd_Q_values_old+self.prd_value_clip),prd_Q_value_target_normalized)

				self.shared_critic_network.pop_art.update(shared_Q_value_target)
				shared_Q_value_target_normalized = torch.sum(self.shared_critic_network.pop_art.normalize(shared_Q_value_target)*shared_old_one_hot_actions, dim=-1).unsqueeze(-1) # gives for all possible actions
				shared_critic_loss_1 = F.smooth_l1_loss(shared_Q_value,shared_Q_value_target_normalized)
				shared_critic_loss_2 = F.smooth_l1_loss(torch.clamp(shared_Q_value, shared_Q_values_old-self.shared_value_clip, shared_Q_values_old+self.shared_value_clip),shared_Q_value_target_normalized)
			else:
				prd_critic_loss_1 = F.smooth_l1_loss(prd_Q_value,prd_Q_value_target)
				prd_critic_loss_2 = F.smooth_l1_loss(torch.clamp(prd_Q_value, prd_Q_values_old-self.prd_value_clip, prd_Q_values_old+self.prd_value_clip),prd_Q_value_target)

				shared_critic_loss_1 = F.smooth_l1_loss(shared_Q_value,shared_Q_value_target)
				shared_critic_loss_2 = F.smooth_l1_loss(torch.clamp(shared_Q_value, shared_Q_values_old-self.shared_value_clip, shared_Q_values_old+self.shared_value_clip),shared_Q_value_target)


			# Finding the ratio (pi_theta / pi_theta__old)
			prd_ratios = torch.exp(prd_logprobs - prd_old_logprobs)
			# Finding Surrogate Loss
			prd_surr1 = prd_ratios * prd_advantage.detach()
			prd_surr2 = torch.clamp(prd_ratios, 1-self.prd_policy_clip, 1+self.prd_policy_clip) * prd_advantage.detach()

			shared_ratios = torch.exp(shared_logprobs - shared_old_logprobs)
			# Finding Surrogate Loss
			shared_surr1 = shared_ratios * shared_advantage.detach()
			shared_surr2 = torch.clamp(shared_ratios, 1-self.shared_policy_clip, 1+self.shared_policy_clip) * shared_advantage.detach()

			# final loss of clipped objective PPO
			prd_entropy = -torch.mean(torch.sum(prd_dists * torch.log(torch.clamp(prd_dists, 1e-10,1.0)), dim=2))
			prd_policy_loss = -torch.min(prd_surr1, prd_surr2).mean() - self.prd_entropy_pen*prd_entropy
			
			prd_entropy_weights = -torch.mean(torch.sum(prd_weights_value* torch.log(torch.clamp(prd_weights_value, 1e-10,1.0)), dim=2))
			prd_critic_loss = torch.max(prd_critic_loss_1, prd_critic_loss_2) + self.prd_critic_weight_entropy_pen*prd_entropy_weights

			shared_entropy = -torch.mean(torch.sum(shared_dists * torch.log(torch.clamp(shared_dists, 1e-10,1.0)), dim=2))
			shared_policy_loss = -torch.min(shared_surr1, shared_surr2).mean() - self.shared_entropy_pen*shared_entropy
			
			shared_entropy_weights = -torch.mean(torch.sum(shared_weights_value* torch.log(torch.clamp(shared_weights_value, 1e-10,1.0)), dim=2))
			shared_critic_loss = torch.max(shared_critic_loss_1, shared_critic_loss_2) + self.shared_critic_weight_entropy_pen*shared_entropy_weights
			

			# take gradient step
			self.prd_critic_optimizer.zero_grad()
			prd_critic_loss.backward()
			prd_grad_norm_value = torch.nn.utils.clip_grad_norm_(self.prd_critic_network.parameters(),self.prd_grad_clip_critic)
			self.prd_critic_optimizer.step()

			self.prd_policy_optimizer.zero_grad()
			prd_policy_loss.backward()
			prd_grad_norm_policy = torch.nn.utils.clip_grad_norm_(self.prd_policy_network.parameters(),self.prd_grad_clip_actor)
			self.prd_policy_optimizer.step()

			self.shared_critic_optimizer.zero_grad()
			shared_critic_loss.backward()
			shared_grad_norm_value = torch.nn.utils.clip_grad_norm_(self.shared_critic_network.parameters(),self.shared_grad_clip_critic)
			self.shared_critic_optimizer.step()

			self.shared_policy_optimizer.zero_grad()
			shared_policy_loss.backward()
			shared_grad_norm_policy = torch.nn.utils.clip_grad_norm_(self.shared_policy_network.parameters(),self.shared_grad_clip_actor)
			self.shared_policy_optimizer.step()

			prd_value_loss_batch += prd_critic_loss
			shared_value_loss_batch += shared_critic_loss
			prd_policy_loss_batch += prd_policy_loss
			shared_policy_loss_batch += shared_policy_loss
			prd_entropy_batch += prd_entropy
			shared_entropy_batch += shared_entropy
			prd_grad_norm_value_batch += prd_grad_norm_value
			shared_grad_norm_value_batch += shared_grad_norm_value
			prd_grad_norm_policy_batch += prd_grad_norm_policy
			shared_grad_norm_policy_batch += shared_grad_norm_policy
			if prd_value_weights_batch is None:
				prd_value_weights_batch = torch.zeros_like(prd_weights_value.cpu())
			prd_value_weights_batch += prd_weights_value.detach().cpu()
			if shared_value_weights_batch is None:
				shared_value_weights_batch = torch.zeros_like(shared_weights_value.cpu())
			shared_value_weights_batch += shared_weights_value.detach().cpu()


			
		# Copy new weights into old policy
		self.prd_policy_network_old.load_state_dict(self.prd_policy_network.state_dict())
		self.shared_policy_network_old.load_state_dict(self.prd_policy_network.state_dict())

		# Copy new weights into old critic
		self.prd_critic_network_old.load_state_dict(self.prd_critic_network.state_dict())
		self.shared_critic_network_old.load_state_dict(self.shared_critic_network.state_dict())

		# self.scheduler.step()
		# print("learning rate of policy", self.scheduler.get_lr())

		# clear buffer
		self.buffer.clear()

		prd_value_loss_batch /= self.n_epochs
		shared_value_loss_batch /= self.n_epochs
		prd_policy_loss_batch /= self.n_epochs
		shared_policy_loss_batch /= self.n_epochs
		prd_entropy_batch /= self.n_epochs
		shared_entropy_batch /= self.n_epochs
		prd_grad_norm_value_batch /= self.n_epochs
		shared_grad_norm_value_batch /= self.n_epochs
		prd_grad_norm_policy_batch /= self.n_epochs
		shared_grad_norm_policy_batch /= self.n_epochs
		prd_value_weights_batch /= self.n_epochs
		shared_value_weights_batch /= self.n_epochs
		prd_agent_groups_over_episode_batch /= self.n_epochs
		prd_avg_agent_group_over_episode_batch /= self.n_epochs

		self.update_parameters()


		self.plotting_dict = {
		"prd_value_loss": prd_value_loss_batch,
		"shared_value_loss": shared_value_loss_batch,
		"prd_policy_loss": prd_policy_loss_batch,
		"shared_policy_loss": shared_policy_loss,
		"prd_entropy": prd_entropy_batch,
		"shared_entropy": shared_entropy_batch,
		"prd_grad_norm_value":prd_grad_norm_value_batch,
		"shared_grad_norm_value":shared_grad_norm_value_batch,
		"prd_grad_norm_policy": prd_grad_norm_policy_batch,
		"shared_grad_norm_policy": shared_grad_norm_policy_batch,
		"prd_weights_value": prd_value_weights_batch,
		"shared_weights_value": shared_value_weights_batch,
		}

		if "threshold" in self.prd_type:
			self.plotting_dict["prd_agent_groups_over_episode"] = prd_agent_groups_over_episode_batch
			self.plotting_dict["prd_avg_agent_group_over_episode"] = prd_avg_agent_group_over_episode_batch
		if "prd_top" in self.prd_type:
			self.plotting_dict["prd_mean_min_weight_value"] = prd_mean_min_weight_value

		if self.comet_ml is not None:
			self.plot(episode)