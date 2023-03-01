import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from model import MLP_Policy, Q_V_network
from utils import RolloutBuffer

class PPOAgent:

	def __init__(
		self, 
		env, 
		dictionary,
		comet_ml,
		):

		# Environment Setup
		self.env = env
		self.team_size = dictionary["team_size"]
		self.env_name = dictionary["env"]
		self.num_agents = self.env.n
		self.num_actions = self.env.action_space[0].n

		# Training setup
		self.test_num = dictionary["test_num"]
		self.gif = dictionary["gif"]
		self.experiment_type = dictionary["experiment_type"]
		self.num_relevant_agents_in_relevant_set = []
		self.num_non_relevant_agents_in_relevant_set = []
		self.false_positive_rate = []
		self.n_epochs = dictionary["n_epochs"]
		self.scheduler_need = dictionary["scheduler_need"]
		if dictionary["device"] == "gpu":
			self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		else:
			self.device = "cpu"
		assert self.num_agents%self.team_size == 0
		self.num_teams = self.num_agents//self.team_size
		self.relevant_set = torch.zeros(1,self.num_agents,self.num_agents).to(self.device)

		for team_id in range(self.num_teams):
			for i in range(self.num_agents):
				if i >= team_id*self.team_size and i < (team_id+1)*self.team_size:
					self.relevant_set[0][i][team_id*self.team_size:(team_id+1)*self.team_size] = torch.ones(self.team_size)

		self.non_relevant_set = torch.ones(1,self.num_agents,self.num_agents).to(self.device) - self.relevant_set

		# Critic Setup
		self.value_lr = dictionary["value_lr"]
		self.critic_weight_entropy_pen = dictionary["critic_weight_entropy_pen"]
		self.lambda_ = dictionary["lambda"] # TD lambda
		self.value_clip = dictionary["value_clip"]
		self.num_heads = dictionary["num_heads"]
		self.enable_hard_attention = dictionary["enable_hard_attention"]
		self.grad_clip_critic = dictionary["grad_clip_critic"]


		# Actor Setup
		self.policy_lr = dictionary["policy_lr"]
		self.update_learning_rate_with_prd = dictionary["update_learning_rate_with_prd"]
		self.gamma = dictionary["gamma"]
		self.entropy_pen = dictionary["entropy_pen"]
		self.gae_lambda = dictionary["gae_lambda"]
		self.top_k = dictionary["top_k"]
		self.norm_adv = dictionary["norm_adv"]
		self.norm_returns = dictionary["norm_returns"]
		self.threshold_min = dictionary["threshold_min"]
		self.threshold_max = dictionary["threshold_max"]
		self.steps_to_take = dictionary["steps_to_take"]
		self.policy_clip = dictionary["policy_clip"]
		self.grad_clip_actor = dictionary["grad_clip_actor"]
		if self.enable_hard_attention:
			self.select_above_threshold = 0
		else:
			self.select_above_threshold = dictionary["select_above_threshold"]
			if "prd_above_threshold_decay" in self.experiment_type:
				self.threshold_delta = (self.select_above_threshold - self.threshold_min)/self.steps_to_take
			elif "prd_above_threshold_ascend" in self.experiment_type:
				self.threshold_delta = (self.threshold_max - self.select_above_threshold)/self.steps_to_take


		print("EXPERIMENT TYPE", self.experiment_type)
		obs_input_dim = 2*3+1 # crossing_team_greedy
		# Q-V Network
		self.critic_network = Q_V_network(obs_input_dim=obs_input_dim, num_heads=self.num_heads, num_agents=self.num_agents, num_actions=self.num_actions, device=self.device, enable_hard_attention=self.enable_hard_attention).to(self.device)
		self.critic_network_old = Q_V_network(obs_input_dim=obs_input_dim, num_heads=self.num_heads, num_agents=self.num_agents, num_actions=self.num_actions, device=self.device, enable_hard_attention=self.enable_hard_attention).to(self.device)
		# Copy network params
		self.critic_network_old.load_state_dict(self.critic_network.state_dict())
		# Disable updates for old network
		for param in self.critic_network_old.parameters():
			param.requires_grad_(False)

		self.history_states_critic = None
		
		
		# Policy Network
		obs_input_dim = 2*3+1 + (self.num_agents-1)*(2*2+1) # crossing_team_greedy
		self.policy_network = MLP_Policy(obs_input_dim=obs_input_dim, num_agents=self.num_agents, num_actions=self.num_actions, device=self.device).to(self.device)
		self.policy_network_old = MLP_Policy(obs_input_dim=obs_input_dim, num_agents=self.num_agents, num_actions=self.num_actions, device=self.device).to(self.device)
		# Copy network params
		self.policy_network_old.load_state_dict(self.policy_network.state_dict())
		# Disable updates for old network
		for param in self.policy_network_old.parameters():
			param.requires_grad_(False)
		
		self.buffer = RolloutBuffer()

		# Loading models
		if dictionary["load_models"]:
			# For CPU
			if torch.cuda.is_available() is False:
				self.critic_network.load_state_dict(torch.load(dictionary["model_path_value"], map_location=torch.device('cpu')))
				self.policy_network.load_state_dict(torch.load(dictionary["model_path_policy"], map_location=torch.device('cpu')))
			# For GPU
			else:
				self.critic_network.load_state_dict(torch.load(dictionary["model_path_value"]))
				self.policy_network.load_state_dict(torch.load(dictionary["model_path_policy"]))

		
		self.critic_optimizer = optim.Adam(self.critic_network.parameters(),lr=self.value_lr)
		self.policy_optimizer = optim.Adam(self.policy_network.parameters(),lr=self.policy_lr)

		if self.scheduler_need:
			self.scheduler = optim.lr_scheduler.MultiStepLR(self.policy_optimizer, milestones=[1000, 20000], gamma=0.1)
			self.scheduler = optim.lr_scheduler.MultiStepLR(self.policy_optimizer, milestones=[1000, 20000], gamma=0.1)

		self.comet_ml = None
		if dictionary["save_comet_ml_plot"]:
			self.comet_ml = comet_ml


	def get_action(self, state_policy, greedy=False):
		with torch.no_grad():
			state_policy = torch.FloatTensor(state_policy).to(self.device)
			dists = self.policy_network_old(state_policy)
			if greedy:
				actions = [dist.argmax().detach().cpu().item() for dist in dists]
			else:
				actions = [Categorical(dist).sample().detach().cpu().item() for dist in dists]

			probs = Categorical(dists)
			action_logprob = probs.log_prob(torch.FloatTensor(actions).to(self.device))

			self.buffer.logprobs.append(action_logprob.detach().cpu())

			return actions


	# def calculate_advantages(self, values, rewards, dones):
	# 	advantages = []
	# 	next_value = 0
	# 	advantage = 0
	# 	rewards = rewards.unsqueeze(-1)
	# 	dones = dones.unsqueeze(-1)
	# 	masks = 1 - dones
	# 	for t in reversed(range(0, len(rewards))):
	# 		td_error = rewards[t] + (self.gamma * next_value * masks[t]) - values.data[t]
	# 		next_value = values.data[t]
			
	# 		advantage = td_error + (self.gamma * self.gae_lambda * advantage * masks[t])
	# 		advantages.insert(0, advantage)

	# 	advantages = torch.stack(advantages)
		
	# 	if self.norm_adv:
	# 		advantages = (advantages - advantages.mean()) / advantages.std()
		
	# 	return advantages



	def calculate_deltas(self, values, rewards, dones):
		deltas = []
		next_value = 0
		# rewards = rewards.unsqueeze(-1)
		# dones = dones.unsqueeze(-1)
		masks = 1-dones
		for t in reversed(range(0, len(rewards))):
			td_error = rewards[t] + (self.gamma * next_value * masks[t]) - values.data[t]
			next_value = values.data[t]
			deltas.insert(0,td_error)
		deltas = torch.stack(deltas)

		return deltas


	def nstep_returns(self, values, rewards, dones):
		deltas = self.calculate_deltas(values, rewards, dones)
		advs = self.calculate_returns(deltas, self.gamma*self.lambda_)
		target_Vs = advs+values
		return target_Vs


	def calculate_returns(self, rewards, discount_factor):
		returns = []
		R = 0
		
		for r in reversed(rewards):
			R = r + R * discount_factor
			returns.insert(0, R)
		
		returns_tensor = torch.stack(returns)
		
		if self.norm_returns:
			returns_tensor = (returns_tensor - returns_tensor.mean()) / returns_tensor.std()
			
		return returns_tensor


	def plot(self, episode):
		self.comet_ml.log_metric('Q_Value_Loss',self.plotting_dict["q_value_loss"].item(),episode)
		self.comet_ml.log_metric('V_Value_Loss',self.plotting_dict["v_value_loss"].item(),episode)
		self.comet_ml.log_metric('Grad_Norm_Value',self.plotting_dict["grad_norm_value"],episode)
		self.comet_ml.log_metric('Policy_Loss',self.plotting_dict["policy_loss"].item(),episode)
		self.comet_ml.log_metric('Grad_Norm_Policy',self.plotting_dict["grad_norm_policy"],episode)
		self.comet_ml.log_metric('Entropy',self.plotting_dict["entropy"].item(),episode)

		if "threshold" in self.experiment_type:
			for i in range(self.num_agents):
				agent_name = "agent"+str(i)
				self.comet_ml.log_metric('Group_Size_'+agent_name, self.plotting_dict["agent_groups_over_episode"][i].item(), episode)

			self.comet_ml.log_metric('Avg_Group_Size', self.plotting_dict["avg_agent_group_over_episode"].item(), episode)

			if "crossing_team_greedy" == self.env_name:
				self.comet_ml.log_metric('Num_relevant_agents_in_relevant_set',torch.mean(self.plotting_dict["num_relevant_agents_in_relevant_set"]),episode)
				self.comet_ml.log_metric('Num_non_relevant_agents_in_relevant_set',torch.mean(self.plotting_dict["num_non_relevant_agents_in_relevant_set"]),episode)
				self.num_relevant_agents_in_relevant_set.append(torch.mean(self.plotting_dict["num_relevant_agents_in_relevant_set"]).item())
				self.num_non_relevant_agents_in_relevant_set.append(torch.mean(self.plotting_dict["num_non_relevant_agents_in_relevant_set"]).item())
				# FPR = FP / (FP+TN)
				FP = torch.mean(self.plotting_dict["num_non_relevant_agents_in_relevant_set"]).item()*self.num_agents
				TN = torch.mean(self.plotting_dict["true_negatives"]).item()*self.num_agents
				self.false_positive_rate.append(FP/(FP+TN))

		if "prd_top" in self.experiment_type:
			self.comet_ml.log_metric('Mean_Smallest_Weight', self.plotting_dict["mean_min_weight_value"].item(), episode)


		# ENTROPY OF WEIGHTS
		for i in range(self.num_heads):
			entropy_weights = -torch.mean(torch.sum(self.plotting_dict["weights_prd"][:,i]* torch.log(torch.clamp(self.plotting_dict["weights_prd"][:,i], 1e-10,1.0)), dim=2))
			self.comet_ml.log_metric('Critic_Weight_Entropy_Head_'+str(i+1), entropy_weights.item(), episode)


	# def calculate_advantages_based_on_exp(self, V_values, rewards, dones, weights_prd, episode):
	# 	advantage = None
	# 	masking_advantage = None
	# 	mean_min_weight_value = -1
	# 	if "shared" in self.experiment_type:
	# 		advantage = torch.sum(self.calculate_advantages(V_values, rewards, dones),dim=-2)
	# 	elif "prd_above_threshold" in self.experiment_type:
	# 		masking_advantage = (weights_prd>self.select_above_threshold).int()
	# 		advantage = torch.sum(self.calculate_advantages(V_values, rewards, dones) * torch.transpose(masking_advantage,-1,-2),dim=-2)
	# 	elif "top" in self.experiment_type:
	# 		if episode < self.steps_to_take:
	# 			advantage = torch.sum(self.calculate_advantages(V_values, rewards, dones),dim=-2)
	# 			masking_advantage = torch.ones(weights_prd.shape).to(self.device)
	# 			min_weight_values, _ = torch.min(weights_prd, dim=-1)
	# 			mean_min_weight_value = torch.mean(min_weight_values)
	# 		else:
	# 			values, indices = torch.topk(weights_prd,k=self.top_k,dim=-1)
	# 			min_weight_values, _ = torch.min(values, dim=-1)
	# 			mean_min_weight_value = torch.mean(min_weight_values)
	# 			masking_advantage = torch.sum(F.one_hot(indices, num_classes=self.num_agents), dim=-2)
	# 			advantage = torch.sum(self.calculate_advantages(V_values, rewards, dones) * torch.transpose(masking_advantage,-1,-2),dim=-2)
	# 	elif "greedy" in self.experiment_type:
	# 		advantage = torch.sum(self.calculate_advantages(V_values, rewards, dones) * self.greedy_policy ,dim=-2)
	# 	elif "relevant_set" in self.experiment_type:
	# 		advantage = torch.sum(self.calculate_advantages(V_values, rewards, dones) * self.relevant_set ,dim=-2)

	# 	if "scaled" in self.experiment_type and episode > self.steps_to_take and "top" in self.experiment_type:
	# 		advantage = advantage*(self.num_agents/self.top_k)

	# 	return advantage, masking_advantage, mean_min_weight_value


	def calculate_advantages_Q_V(self, Q, V, weights_prd, dones, episode):
		'''
		Q : B x N x 1
		V : B x N x 1
		weights_prd: B x N x N x 1
		dones: B x N x 1
		episode : int
		'''
		advantage = None
		masking_advantage = None
		mean_min_weight_value = -1
		dones = dones.unsqueeze(-1)

		if "shared" in self.experiment_type:
			advantage = torch.sum((Q - V).unsqueeze(1).repeat(1, self.num_agents, 1) * (1-dones), dim=-2) # B x N x 1

		elif "prd_above_threshold" in self.experiment_type:
			masking_advantage = (weights_prd > self.select_above_threshold).int()
			advantage = torch.sum((Q - V).unsqueeze(1).repeat(1, self.num_agents, 1) * torch.transpose(masking_advantage,-1,-2) * (1-dones), dim=-2) # B x N x 1
	
		elif "top" in self.experiment_type:
			# No masking until warm-up period ends
			if episode < self.steps_to_take:
				advantage = torch.sum((Q - V).unsqueeze(1).repeat(1, self.num_agents, 1) * torch.transpose(masking_advantage,-1,-2) * (1-dones), dim=-2) # B x N x 1
				masking_advantage = torch.ones(weights_prd.shape).to(self.device)
				min_weight_values, _ = torch.min(weights_prd, dim=-1)
				mean_min_weight_value = torch.mean(min_weight_values)
			else:
				values, indices = torch.topk(weights_prd,k=self.top_k,dim=-1)
				min_weight_values, _ = torch.min(values, dim=-1)
				mean_min_weight_value = torch.mean(min_weight_values)
				masking_advantage = torch.sum(F.one_hot(indices, num_classes=self.num_agents), dim=-2)
				advantage = torch.sum((Q - V).unsqueeze(1).repeat(1, self.num_agents, 1, 1) * torch.transpose(masking_advantage,-1,-2) * (1-dones), dim=-2) # B x N x 1
		
		if "scaled" in self.experiment_type and episode > self.steps_to_take and "top" in self.experiment_type:
			advantage = advantage*(self.num_agents/self.top_k)

		return advantage.detach(), masking_advantage, mean_min_weight_value

	def update_parameters(self):
		if self.select_above_threshold > self.threshold_min and "prd_above_threshold_decay" in self.experiment_type:
			self.select_above_threshold = self.select_above_threshold - self.threshold_delta

		if self.threshold_max > self.select_above_threshold and "prd_above_threshold_ascend" in self.experiment_type:
			self.select_above_threshold = self.select_above_threshold + self.threshold_delta



	def update(self,episode):
		# convert list to tensor
		old_states_critic = torch.FloatTensor(np.array(self.buffer.states_critic))
		# history_states_critic = torch.stack(self.buffer.history_states_critic, dim=0)
		# Q_values_old = torch.stack(self.buffer.Q_values, dim=0)
		# Values_old = torch.stack(self.buffer.Values, dim=0)
		old_states_actor = torch.FloatTensor(np.array(self.buffer.states_actor))
		old_actions = torch.FloatTensor(np.array(self.buffer.actions))
		old_one_hot_actions = torch.FloatTensor(np.array(self.buffer.one_hot_actions))
		old_logprobs = torch.stack(self.buffer.logprobs, dim=0)
		rewards = torch.FloatTensor(np.array(self.buffer.rewards))
		dones = torch.FloatTensor(np.array(self.buffer.dones)).long()

		if self.history_states_critic is None:
			self.history_states_critic = torch.zeros(old_states_critic.shape[0], self.num_agents, 256)

		with torch.no_grad():
			Q_values_old, Values_old, self.history_states_critic, _ = self.critic_network_old(
																	old_states_critic.to(self.device),
																	self.history_states_critic.to(self.device),
																	old_one_hot_actions.to(self.device)
																	)

		Q_value_target = self.nstep_returns(Q_values_old.cpu(), rewards, dones).to(self.device)

		q_value_loss_batch = 0
		v_value_loss_batch = 0
		policy_loss_batch = 0
		entropy_batch = 0
		weight_prd_batch = None
		grad_norm_value_batch = 0
		grad_norm_policy_batch = 0
		agent_groups_over_episode_batch = 0
		avg_agent_group_over_episode_batch = 0

		# torch.autograd.set_detect_anomaly(True)
		# Optimize policy for n epochs
		for _ in range(self.n_epochs):

			Q_value, Value, new_history_states_critic, weights_prd = self.critic_network(old_states_critic.to(self.device), self.history_states_critic.to(self.device), old_one_hot_actions.to(self.device))

			advantage, masking_advantage, mean_min_weight_value = self.calculate_advantages_Q_V(Q_value, Value, torch.mean(weights_prd.detach(), dim=1), dones.to(self.device), episode)

			dists = self.policy_network(old_states_actor.to(self.device))
			probs = Categorical(dists.squeeze(0))
			logprobs = probs.log_prob(old_actions.to(self.device))

			if "threshold" in self.experiment_type:
				agent_groups_over_episode = torch.sum(torch.sum(masking_advantage.float(), dim=-2),dim=0)/masking_advantage.shape[0]
				avg_agent_group_over_episode = torch.mean(agent_groups_over_episode)
				agent_groups_over_episode_batch += agent_groups_over_episode
				avg_agent_group_over_episode_batch += avg_agent_group_over_episode

				target_V_rewards = torch.sum(rewards.unsqueeze(-2).repeat(1, self.num_agents, 1) * torch.transpose(masking_advantage.detach().cpu(),-1,-2), dim=-1)
				Value_target = self.nstep_returns(Values_old, target_V_rewards, dones).to(self.device)
			else:
				target_V_rewards = torch.sum(rewards.unsqueeze(-2).repeat(1, self.num_agents, 1), dim=-1)
				Value_target = self.nstep_returns(Values_old, target_V_rewards, dones).to(self.device)

			critic_v_loss_1 = F.mse_loss(Value, Value_target)
			critic_v_loss_2 = F.mse_loss(torch.clamp(Value, Values_old.to(self.device)-self.value_clip, Values_old.to(self.device)+self.value_clip), Value_target)

			critic_q_loss_1 = F.mse_loss(Q_value, Q_value_target)
			critic_q_loss_2 = F.mse_loss(torch.clamp(Q_value, Q_values_old.to(self.device)-self.value_clip, Q_values_old.to(self.device)+self.value_clip), Q_value_target)

			# Finding the ratio (pi_theta / pi_theta__old)
			ratios = torch.exp(logprobs - old_logprobs.to(self.device))
			# Finding Surrogate Loss
			surr1 = ratios * advantage
			surr2 = torch.clamp(ratios, 1-self.policy_clip, 1+self.policy_clip) * advantage

			# final loss of clipped objective PPO
			entropy = -torch.mean(torch.sum(dists * torch.log(torch.clamp(dists, 1e-10,1.0)), dim=2))
			policy_loss = -torch.min(surr1, surr2).mean() - self.entropy_pen*entropy
			
			entropy_weights = 0
			for i in range(self.num_heads):
				entropy_weights += -torch.mean(torch.sum(weights_prd[:, i] * torch.log(torch.clamp(weights_prd[:, i], 1e-10,1.0)), dim=2))

			critic_q_loss = torch.max(critic_q_loss_1, critic_q_loss_2) + self.critic_weight_entropy_pen*entropy_weights
			critic_v_loss = torch.max(critic_v_loss_1, critic_v_loss_2)
			

			# take gradient step
			self.critic_optimizer.zero_grad()
			(critic_q_loss + critic_v_loss).backward(retain_graph=True)
			grad_norm_value = torch.nn.utils.clip_grad_norm_(self.critic_network.parameters(), self.grad_clip_critic)
			self.critic_optimizer.step()

			self.policy_optimizer.zero_grad()
			policy_loss.backward()
			grad_norm_policy = torch.nn.utils.clip_grad_norm_(self.policy_network.parameters(), self.grad_clip_actor)
			self.policy_optimizer.step()

			self.history_states_critic = new_history_states_critic.detach().cpu()

			q_value_loss_batch += critic_q_loss
			v_value_loss_batch += critic_v_loss
			policy_loss_batch += policy_loss
			entropy_batch += entropy
			grad_norm_value_batch += grad_norm_value
			grad_norm_policy_batch += grad_norm_policy
			if weight_prd_batch is None:
				weight_prd_batch = weights_prd.detach().cpu()
			weight_prd_batch += weights_prd.detach().cpu()


			
		# Copy new weights into old policy
		self.policy_network_old.load_state_dict(self.policy_network.state_dict())

		# Copy new weights into old critic
		self.critic_network_old.load_state_dict(self.critic_network.state_dict())

		# self.scheduler.step()
		# print("learning rate of policy", self.scheduler.get_lr())

		# clear buffer
		self.buffer.clear()

		q_value_loss_batch /= self.n_epochs
		v_value_loss_batch /= self.n_epochs
		policy_loss_batch /= self.n_epochs
		entropy_batch /= self.n_epochs
		grad_norm_value_batch /= self.n_epochs
		grad_norm_policy_batch /= self.n_epochs
		weight_prd_batch /= self.n_epochs
		agent_groups_over_episode_batch /= self.n_epochs
		avg_agent_group_over_episode_batch /= self.n_epochs

		if "prd" in self.experiment_type and "crossing_team_greedy" == self.env_name:
			num_relevant_agents_in_relevant_set = self.relevant_set*masking_advantage
			num_non_relevant_agents_in_relevant_set = self.non_relevant_set*masking_advantage
			true_negatives = self.non_relevant_set*(1-masking_advantage)
		else:
			num_relevant_agents_in_relevant_set = None
			num_non_relevant_agents_in_relevant_set = None
			true_negatives = None


		if "prd" in self.experiment_type:
			if self.update_learning_rate_with_prd:
				for g in self.policy_optimizer.param_groups:
					g['lr'] = self.policy_lr * self.num_agents/avg_agent_group_over_episode_batch

		self.update_parameters()


		self.plotting_dict = {
		"q_value_loss": q_value_loss_batch,
		"v_value_loss": v_value_loss_batch,
		"policy_loss": policy_loss_batch,
		"entropy": entropy_batch,
		"grad_norm_value":grad_norm_value_batch,
		"grad_norm_policy": grad_norm_policy_batch,
		"weights_prd": weight_prd_batch,
		"num_relevant_agents_in_relevant_set": num_relevant_agents_in_relevant_set,
		"num_non_relevant_agents_in_relevant_set": num_non_relevant_agents_in_relevant_set,
		"true_negatives": true_negatives,
		}

		if "threshold" in self.experiment_type:
			self.plotting_dict["agent_groups_over_episode"] = agent_groups_over_episode_batch
			self.plotting_dict["avg_agent_group_over_episode"] = avg_agent_group_over_episode_batch
		if "prd_top" in self.experiment_type:
			self.plotting_dict["mean_min_weight_value"] = mean_min_weight_value

		if self.comet_ml is not None:
			self.plot(episode)

		# return history_states_critic[-1]
