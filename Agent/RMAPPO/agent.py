import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from model import MLP_Policy, Q_network, V_network, ValueNorm, RunningMeanStd
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
		self.env_name = dictionary["env"]
		self.num_agents = self.env.n_agents
		self.num_enemies = self.env.n_enemies
		self.num_actions = self.env.action_space[0].n

		# Training setup
		self.max_episodes = dictionary["max_episodes"]
		self.test_num = dictionary["test_num"]
		self.gif = dictionary["gif"]
		self.experiment_type = dictionary["experiment_type"]
		self.n_epochs = dictionary["n_epochs"]
		self.scheduler_need = dictionary["scheduler_need"]
		self.norm_rewards = dictionary["norm_rewards"]
		if dictionary["device"] == "gpu":
			self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		else:
			self.device = "cpu"

		self.update_ppo_agent = dictionary["update_ppo_agent"]
		self.max_time_steps = dictionary["max_time_steps"]

		# Critic Setup
		self.temperature_v = dictionary["temperature_v"]
		self.temperature_q = dictionary["temperature_q"]
		self.rnn_hidden_q = dictionary["rnn_hidden_q"]
		self.rnn_hidden_v = dictionary["rnn_hidden_v"]
		self.critic_ally_observation = dictionary["ally_observation"]
		self.critic_enemy_observation = dictionary["enemy_observation"]
		self.q_value_lr = dictionary["q_value_lr"]
		self.v_value_lr = dictionary["v_value_lr"]
		self.q_weight_decay = dictionary["q_weight_decay"]
		self.v_weight_decay = dictionary["v_weight_decay"]
		self.critic_weight_entropy_pen = dictionary["critic_weight_entropy_pen"]
		self.critic_weight_entropy_pen_final = dictionary["critic_weight_entropy_pen_final"]
		self.critic_weight_entropy_pen_decay_rate = (dictionary["critic_weight_entropy_pen_final"] - dictionary["critic_weight_entropy_pen"]) / dictionary["critic_weight_entropy_pen_steps"]
		self.critic_score_regularizer = dictionary["critic_score_regularizer"]
		self.lambda_ = dictionary["lambda"] # TD lambda
		self.value_clip = dictionary["value_clip"]
		self.num_heads = dictionary["num_heads"]
		self.enable_hard_attention = dictionary["enable_hard_attention"]
		self.enable_grad_clip_critic = dictionary["enable_grad_clip_critic"]
		self.grad_clip_critic = dictionary["grad_clip_critic"]


		# Actor Setup
		self.warm_up = dictionary["warm_up"]
		self.warm_up_episodes = dictionary["warm_up_episodes"]
		self.epsilon_start = self.epsilon = dictionary["epsilon_start"]
		self.epsilon_end = dictionary["epsilon_end"]
		self.rnn_hidden_actor = dictionary["rnn_hidden_actor"]
		self.actor_observation_shape = dictionary["local_observation"]
		self.policy_lr = dictionary["policy_lr"]
		self.policy_weight_decay = dictionary["policy_weight_decay"]
		self.update_learning_rate_with_prd = dictionary["update_learning_rate_with_prd"]
		self.gamma = dictionary["gamma"]
		self.entropy_pen = dictionary["entropy_pen"]
		self.entropy_pen_decay = (dictionary["entropy_pen"] - dictionary["entropy_pen_final"])/dictionary["entropy_pen_steps"]
		self.entropy_pen_final = dictionary["entropy_pen_final"]
		self.gae_lambda = dictionary["gae_lambda"]
		self.top_k = dictionary["top_k"]
		self.norm_adv = dictionary["norm_adv"]
		self.norm_returns = dictionary["norm_returns"]
		self.threshold_min = dictionary["threshold_min"]
		self.threshold_max = dictionary["threshold_max"]
		self.steps_to_take = dictionary["steps_to_take"]
		self.policy_clip = dictionary["policy_clip"]
		self.enable_grad_clip_actor = dictionary["enable_grad_clip_actor"]
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
		# obs_input_dim = 2*3+1 # crossing_team_greedy
		# Q-V Network
		self.critic_network_q = Q_network(
			ally_obs_input_dim=self.critic_ally_observation, 
			enemy_obs_input_dim=self.critic_enemy_observation, 
			num_heads=self.num_heads, 
			num_agents=self.num_agents, 
			num_enemies=self.num_enemies, 
			num_actions=self.num_actions, 
			device=self.device, 
			enable_hard_attention=self.enable_hard_attention, 
			attention_dropout_prob=dictionary["attention_dropout_prob_q"], 
			temperature=self.temperature_q
			).to(self.device)
		# self.critic_network_q_old = Q_network(
		# 	ally_obs_input_dim=self.critic_ally_observation, 
		# 	enemy_obs_input_dim=self.critic_enemy_observation, 
		# 	num_heads=self.num_heads, 
		# 	num_agents=self.num_agents, 
		# 	num_enemies=self.num_enemies, 
		# 	num_actions=self.num_actions, 
		# 	device=self.device, 
		# 	enable_hard_attention=self.enable_hard_attention, 
		# 	attention_dropout_prob=dictionary["attention_dropout_prob_q"], 
		# 	temperature=self.temperature_q
		# 	).to(self.device)
		# Copy network params
		# self.critic_network_q_old.load_state_dict(self.critic_network_q.state_dict())
		# Disable updates for old network
		# for param in self.critic_network_q_old.parameters():
		# 	param.requires_grad_(False)

		self.critic_network_v = V_network(
			ally_obs_input_dim=self.critic_ally_observation, 
			enemy_obs_input_dim=self.critic_enemy_observation, 
			num_heads=self.num_heads, 
			num_agents=self.num_agents, 
			num_enemies=self.num_enemies, 
			num_actions=self.num_actions, 
			device=self.device, 
			enable_hard_attention=self.enable_hard_attention, 
			attention_dropout_prob=dictionary["attention_dropout_prob_v"], 
			temperature=self.temperature_v
			).to(self.device)
		# self.critic_network_v_old = V_network(
		# 	ally_obs_input_dim=self.critic_ally_observation, 
		# 	enemy_obs_input_dim=self.critic_enemy_observation, 
		# 	num_heads=self.num_heads, 
		# 	num_agents=self.num_agents, 
		# 	num_enemies=self.num_enemies, 
		# 	num_actions=self.num_actions, 
		# 	device=self.device, 
		# 	enable_hard_attention=self.enable_hard_attention, 
		# 	attention_dropout_prob=dictionary["attention_dropout_prob_v"], 
		# 	temperature=self.temperature_v
		# 	).to(self.device)
		# Copy network params
		# self.critic_network_v_old.load_state_dict(self.critic_network_v.state_dict())
		# Disable updates for old network
		# for param in self.critic_network_v_old.parameters():
		# 	param.requires_grad_(False)
		
		
		# Policy Network
		self.policy_network = MLP_Policy(obs_input_dim=self.actor_observation_shape, num_agents=self.num_agents, num_actions=self.num_actions, device=self.device).to(self.device)
		# self.policy_network_old = MLP_Policy(obs_input_dim=self.actor_observation_shape, num_agents=self.num_agents, num_actions=self.num_actions, device=self.device).to(self.device)
		# Copy network params
		# self.policy_network_old.load_state_dict(self.policy_network.state_dict())
		# Disable updates for old network
		# for param in self.policy_network_old.parameters():
		# 	param.requires_grad_(False)

		self.network_update_interval = dictionary["network_update_interval"]
		
		self.buffer = RolloutBuffer(
			num_episodes=self.update_ppo_agent, 
			max_time_steps=self.max_time_steps, 
			num_agents=self.num_agents, 
			num_enemies=self.num_enemies,
			obs_shape_critic_ally=self.critic_ally_observation, 
			obs_shape_critic_enemy=self.critic_enemy_observation, 
			obs_shape_actor=self.actor_observation_shape, 
			num_actions=self.num_actions,
			)

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


		self.q_critic_optimizer = optim.AdamW(self.critic_network_q.parameters(), lr=self.v_value_lr, weight_decay=self.v_weight_decay, eps=1e-05)
		self.v_critic_optimizer = optim.AdamW(self.critic_network_v.parameters(), lr=self.q_value_lr, weight_decay=self.q_weight_decay, eps=1e-05)
		self.policy_optimizer = optim.AdamW(self.policy_network.parameters(),lr=self.policy_lr, weight_decay=self.policy_weight_decay, eps=1e-05)

		if self.scheduler_need:
			self.scheduler_policy = optim.lr_scheduler.MultiStepLR(self.policy_optimizer, milestones=[1000, 20000], gamma=0.1)
			self.scheduler_q_critic = optim.lr_scheduler.MultiStepLR(self.q_critic_optimizer, milestones=[1000, 20000], gamma=0.1)
			self.scheduler_v_critic = optim.lr_scheduler.MultiStepLR(self.v_critic_optimizer, milestones=[1000, 20000], gamma=0.1)

		self.comet_ml = None
		if dictionary["save_comet_ml_plot"]:
			self.comet_ml = comet_ml

		if self.norm_returns:
			self.v_value_norm = ValueNorm(input_shape=1, norm_axes=1, device=self.device)
			self.q_value_norm = ValueNorm(input_shape=1, norm_axes=1, device=self.device)

			# self.v_value_norm = RunningMeanStd(shape=(self.num_agents), device=self.device)
			# self.q_value_norm = RunningMeanStd(shape=(self.num_agents), device=self.device)

		if self.norm_rewards:
			self.reward_norm = RunningMeanStd(shape=(1), device=self.device)


	def get_lr(self, it, learning_rate):
		# 1) linear warmup for warmup_iters steps
		warmup_iters = 250
		lr_decay_iters = 20000
		min_lr = 5e-5
		if it < warmup_iters:
			learning_rate = 5e-4
			return learning_rate * it / warmup_iters
		# 2) if it > lr_decay_iters, return min learning rate
		if it > lr_decay_iters:
			return min_lr
		# 3) in between, use cosine decay down to min learning rate
		decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
		assert 0 <= decay_ratio <= 1
		coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
		return min_lr + coeff * (learning_rate - min_lr)

	def lr_decay(self, episode, initial_lr):
		"""Decreases the learning rate linearly"""
		lr = initial_lr - (initial_lr * (episode / float(self.max_episodes)))
		# for param_group in optimizer.param_groups:
		# 	param_group['lr'] = lr
		return lr

	def get_critic_hidden_state(self, state_allies, state_enemies, one_hot_actions, rnn_hidden_state_q, rnn_hidden_state_v):
		with torch.no_grad():
			state_allies = torch.FloatTensor(state_allies).unsqueeze(0)
			state_enemies = torch.FloatTensor(state_enemies).unsqueeze(0)
			one_hot_actions = torch.FloatTensor(one_hot_actions).unsqueeze(0)
			rnn_hidden_state_q = torch.FloatTensor(rnn_hidden_state_q)
			rnn_hidden_state_v = torch.FloatTensor(rnn_hidden_state_v)
			_, _, _, rnn_hidden_state_v = self.critic_network_v(state_allies.to(self.device), state_enemies.to(self.device), one_hot_actions.to(self.device), rnn_hidden_state_v.to(self.device))
			_, _, _, rnn_hidden_state_q = self.critic_network_q(state_allies.to(self.device), state_enemies.to(self.device), one_hot_actions.to(self.device), rnn_hidden_state_q.to(self.device))

			return V_value.squeeze(0).cpu().numpy(), Q_value.squeeze(0).cpu().numpy(), torch.mean(weights_prd, dim=1).cpu().numpy()

	def update_epsilon(self):
		if self.warm_up:
			self.epsilon -= (self.epsilon_start - self.epsilon_end)/self.warm_up_episodes
			self.epsilon = max(self.epsilon, self.epsilon_end)

	def get_action(self, state_policy, last_one_hot_actions, mask_actions, hidden_state, greedy=False):
		with torch.no_grad():
			state_policy = torch.FloatTensor(state_policy).unsqueeze(1)
			last_one_hot_actions = torch.FloatTensor(last_one_hot_actions).unsqueeze(1)
			final_state_policy = torch.cat([state_policy, last_one_hot_actions], dim=-1).to(self.device)
			mask_actions = torch.BoolTensor(mask_actions).unsqueeze(1).to(self.device)
			hidden_state = torch.FloatTensor(hidden_state).to(self.device)
			dists, hidden_state = self.policy_network(final_state_policy, hidden_state, mask_actions)
			if self.warm_up:
				available_actions = (mask_actions>0).int()
				dists = (1.0-self.epsilon)*dists + available_actions*self.epsilon/torch.sum(available_actions, dim=-1).unsqueeze(-1)
			if greedy:
				actions = [dist.argmax().detach().cpu().item() for dist in dists]
				action_logprob = None
			else:
				actions = [Categorical(dist).sample().detach().cpu().item() for dist in dists]

				probs = Categorical(dists)
				action_logprob = probs.log_prob(torch.FloatTensor(actions).to(self.device)).cpu().numpy()

				# dists_taken = torch.gather(dists, dim=-1, index=torch.LongTensor(actions).unsqueeze(-1).to(self.device))
				# action_logprob = torch.log(dists_taken + 1e-10).squeeze(-1).cpu().numpy()

				# old_pi_taken = th.gather(old_pi, dim=3, index=actions).squeeze(3)
				# old_log_pi_taken = th.log(old_pi_taken + 1e-10)

			return actions, action_logprob, hidden_state.cpu().numpy()


	def calculate_advantages(self, values, next_value, rewards, dones, masks_, next_mask):
		values = values.reshape(self.update_ppo_agent, -1, self.num_agents)
		values_old = values.reshape(self.update_ppo_agent, -1, self.num_agents)
		rewards = rewards.reshape(self.update_ppo_agent, -1, self.num_agents)
		dones = dones.reshape(self.update_ppo_agent, -1, self.num_agents)
		# masks_ = masks_.reshape(self.update_ppo_agent, -1, 1)
		# masks_ = masks_.reshape(self.update_ppo_agent, -1, self.num_agents)
		advantages = rewards.new_zeros(*rewards.shape)
		# next_value = 0
		advantage = 0
		masks = 1 - dones
		for t in reversed(range(0, rewards.shape[1])):
			td_error = rewards[:,t,:] + (self.gamma * next_value * next_mask) - values.data[:,t,:]
			advantage = (td_error + (self.gamma * self.gae_lambda * advantage * next_mask)) * next_mask #*masks_[:,t,:]
			# print("mask")
			# print(next_mask)
			# print("advantage")
			# print(advantage)
			next_value = values.data[:, t, :]
			next_mask = masks[:,t,:]
			
			advantages[:,t,:] = advantage

		# advantages = torch.stack(advantages)
		return advantages.reshape(-1, self.num_agents)



	def build_td_lambda_targets(self, rewards, terminated, mask, target_qs):
		# Assumes  <target_qs > in B*T*A and <reward >, <terminated >  in B*T*A, <mask > in (at least) B*T-1*1
		# Initialise  last  lambda -return  for  not  terminated  episodes
		ret = target_qs.new_zeros(*target_qs.shape)
		ret = target_qs * (1-terminated[:, :-1]) # some episodes end early so we can't assume that by copying the last target_qs in ret would be good enough
		# ret[:, -1] = target_qs[:, -1] * (1 - (torch.sum(terminated, dim=1)>0).int())
		# Backwards  recursive  update  of the "forward  view"
		for t in range(ret.shape[1] - 2, -1, -1):
			ret[:, t] = self.lambda_ * self.gamma * ret[:, t + 1] + mask[:, t].unsqueeze(-1) \
						* (rewards[:, t] + (1 - self.lambda_) * self.gamma * target_qs[:, t + 1] * (1 - terminated[:, t]))
		# Returns lambda-return from t=0 to t=T-1, i.e. in B*T-1*A
		# return ret[:, 0:-1]
		return ret



	def plot(self, masks, episode):
		self.comet_ml.log_metric('Q_Value_Loss',self.plotting_dict["q_value_loss"],episode)
		self.comet_ml.log_metric('V_Value_Loss',self.plotting_dict["v_value_loss"],episode)
		self.comet_ml.log_metric('Grad_Norm_V_Value',self.plotting_dict["grad_norm_value_v"],episode)
		self.comet_ml.log_metric('Grad_Norm_Q_Value',self.plotting_dict["grad_norm_value_q"],episode)
		self.comet_ml.log_metric('Policy_Loss',self.plotting_dict["policy_loss"],episode)
		self.comet_ml.log_metric('Grad_Norm_Policy',self.plotting_dict["grad_norm_policy"],episode)
		self.comet_ml.log_metric('Entropy',self.plotting_dict["entropy"],episode)

		self.comet_ml.log_metric('Q_Value_LR',self.plotting_dict["q_value_lr"],episode)
		self.comet_ml.log_metric('V_Value_LR',self.plotting_dict["v_value_lr"],episode)
		self.comet_ml.log_metric('Policy_LR',self.plotting_dict["policy_lr"],episode)

		if "threshold" in self.experiment_type:
			for i in range(self.num_agents):
				agent_name = "agent"+str(i)
				self.comet_ml.log_metric('Group_Size_'+agent_name, self.plotting_dict["agent_groups_over_episode"][i].item(), episode)

			self.comet_ml.log_metric('Avg_Group_Size', self.plotting_dict["avg_agent_group_over_episode"].item(), episode)

		# ENTROPY OF Q WEIGHTS
		for i in range(self.num_heads):
			# entropy_weights = -torch.mean(torch.sum(self.plotting_dict["weights_prd"][:,i]* torch.log(torch.clamp(self.plotting_dict["weights_prd"][:,i], 1e-10,1.0)), dim=-1))
			entropy_weights = -torch.sum(torch.sum((self.plotting_dict["weights_prd"][:, i] * torch.log(torch.clamp(self.plotting_dict["weights_prd"][:, i], 1e-10, 1.0)) * masks.unsqueeze(-1)), dim=-1))/masks.sum() #(masks.sum()*self.num_agents)
			self.comet_ml.log_metric('Q_Weight_Entropy_Head_'+str(i+1), entropy_weights.item(), episode)

		# ENTROPY OF V WEIGHTS
		for i in range(self.num_heads):
			# entropy_weights = -torch.mean(torch.sum(self.plotting_dict["weights_v"][:,i]* torch.log(torch.clamp(self.plotting_dict["weights_v"][:,i], 1e-10,1.0)), dim=-1))
			entropy_weights = -torch.sum(torch.sum((self.plotting_dict["weights_v"][:, i] * torch.log(torch.clamp(self.plotting_dict["weights_v"][:, i], 1e-10, 1.0)) * masks.unsqueeze(-1)), dim=-1))/masks.sum() #(masks.sum()*self.num_agents)
			self.comet_ml.log_metric('V_Weight_Entropy_Head_'+str(i+1), entropy_weights.item(), episode)


	def calculate_advantages_based_on_exp(self, V_values, next_Values, rewards, dones, weights_prd, masks, next_mask, episode):
		advantage = None
		masking_rewards = None
		mean_min_weight_value = -1
		if "shared" in self.experiment_type:
			rewards_ = torch.sum(rewards.unsqueeze(-2).repeat(1, self.num_agents, 1), dim=-1)
			advantage = self.calculate_advantages(V_values, next_Values, rewards_, dones, masks, next_mask)
		elif "prd_above_threshold_ascend" in self.experiment_type or "prd_above_threshold_decay" in self.experiment_type:
			masking_rewards = (weights_prd>self.select_above_threshold).int()
			rewards_ = torch.sum(rewards.unsqueeze(-2).repeat(1, self.num_agents, 1) * torch.transpose(masking_rewards,-1,-2), dim=-1)
			advantage = self.calculate_advantages(V_values, next_Values, rewards_, dones, masks, next_mask)
		elif "prd_above_threshold" in self.experiment_type:
			if episode > self.steps_to_take:
				masking_rewards = (weights_prd>self.select_above_threshold).int()
				rewards_ = torch.sum(rewards.unsqueeze(-2).repeat(1, self.num_agents, 1) * torch.transpose(masking_rewards,-1,-2), dim=-1)
			else:
				masking_rewards = torch.ones(weights_prd.shape).to(self.device)
				rewards_ = torch.sum(rewards.unsqueeze(-2).repeat(1, self.num_agents, 1), dim=-1)
			advantage = self.calculate_advantages(V_values, next_Values, rewards_, dones, masks, next_mask)
		elif "top" in self.experiment_type:
			if episode > self.steps_to_take:
				rewards_ = torch.sum(rewards.unsqueeze(-2).repeat(1, self.num_agents, 1), dim=-1)
				advantage = self.calculate_advantages(V_values, next_Values, rewards_, dones, masks, next_mask)
				masking_rewards = torch.ones(weights_prd.shape).to(self.device)
				min_weight_values, _ = torch.min(weights_prd, dim=-1)
				mean_min_weight_value = torch.mean(min_weight_values)
			else:
				values, indices = torch.topk(weights_prd,k=self.top_k,dim=-1)
				min_weight_values, _ = torch.min(values, dim=-1)
				mean_min_weight_value = torch.mean(min_weight_values)
				masking_rewards = torch.sum(F.one_hot(indices, num_classes=self.num_agents), dim=-2)
				rewards_ = torch.sum(masking_rewards * torch.transpose(masking_rewards,-1,-2), dim=-1)
				advantage = self.calculate_advantages(V_values, next_Values, rewards_, dones, masks, next_mask)
		elif "prd_soft_advantage" in self.experiment_type:
			if episode > self.steps_to_take:
				rewards_ = torch.sum(rewards.unsqueeze(-2).repeat(1, self.num_agents, 1) * torch.transpose(weights_prd, -1, -2), dim=-1)
			else:
				rewards_ = torch.sum(rewards.unsqueeze(-2).repeat(1, self.num_agents, 1), dim=-1)
			
			advantage = self.calculate_advantages(V_values, next_Values, rewards_, dones, masks, next_mask)
		
		if "scaled" in self.experiment_type and episode > self.steps_to_take and "top" in self.experiment_type:
			advantage *= self.num_agents
			if "top" in self.experiment_type:
				advantage /= self.top_k


		return advantage.detach(), masking_rewards, mean_min_weight_value


	def update_parameters(self):
		if self.select_above_threshold > self.threshold_min and "prd_above_threshold_decay" in self.experiment_type:
			self.select_above_threshold = self.select_above_threshold - self.threshold_delta

		if self.threshold_max > self.select_above_threshold and "prd_above_threshold_ascend" in self.experiment_type:
			self.select_above_threshold = self.select_above_threshold + self.threshold_delta

		if self.critic_weight_entropy_pen_final + self.critic_weight_entropy_pen_decay_rate > self.critic_weight_entropy_pen:
			self.critic_weight_entropy_pen += self.critic_weight_entropy_pen_decay_rate 

		if self.entropy_pen - self.entropy_pen_decay > self.entropy_pen_final:
			self.entropy_pen -= self.entropy_pen_decay

	def update(self, episode):
		# convert list to tensor
		# BATCH, TIMESTEPS, NUM AGENTS, DIM
		old_states_critic_allies = torch.FloatTensor(np.array(self.buffer.states_critic_allies))[:, :-1, :, :]
		old_states_critic_enemies = torch.FloatTensor(np.array(self.buffer.states_critic_enemies))[:, :-1, :, :]
		old_states_actor = torch.FloatTensor(np.array(self.buffer.states_actor))
		old_actions = torch.FloatTensor(np.array(self.buffer.actions)).reshape(-1, self.num_agents)
		old_last_one_hot_actions = torch.FloatTensor(np.array(self.buffer.last_one_hot_actions))
		old_one_hot_actions = torch.FloatTensor(np.array(self.buffer.one_hot_actions))[:, :-1, :, :]
		old_mask_actions = torch.BoolTensor(np.array(self.buffer.action_masks)).reshape(-1, self.num_agents, self.num_actions)
		old_logprobs = torch.FloatTensor(self.buffer.logprobs).reshape(-1, self.num_agents)
		rewards = torch.FloatTensor(np.array(self.buffer.rewards)).reshape(-1, self.num_agents)
		dones = torch.FloatTensor(np.array(self.buffer.dones))[:, :-1, :].long().reshape(-1, self.num_agents)
		# masks = torch.FloatTensor(np.array(self.buffer.masks)).long()
		masks = 1 - dones.reshape(self.update_ppo_agent, -1, self.num_agents)

		if self.norm_rewards:
			self.reward_norm.update(rewards.view(-1).to(self.device), masks.view(-1).to(self.device))
			rewards = ((rewards.to(self.device) - self.reward_norm.mean) / (torch.sqrt(self.reward_norm.var) + 1e-5)).cpu().view(-1, self.num_agents)
		
		# if episode < 25:
		# 	self.buffer.clear()
		# 	return

		# batch, _, _ = masks.shape
		rnn_hidden_state_q = torch.zeros(1, self.update_ppo_agent*self.num_agents, self.rnn_hidden_q)
		rnn_hidden_state_v = torch.zeros(1, self.update_ppo_agent*self.num_agents, self.rnn_hidden_v)
		rnn_hidden_state_actor = torch.zeros(1, self.update_ppo_agent*self.num_agents, self.rnn_hidden_actor)

		max_episode_len = int(np.max(self.buffer.episode_length))

		v_value_lr, q_value_lr, policy_lr = self.v_value_lr, self.q_value_lr, self.policy_lr
		# v_value_lr = self.lr_decay(episode, self.v_value_lr)
		# for param_group in self.v_critic_optimizer.param_groups:
		# 	param_group['lr'] = v_value_lr

		# q_value_lr = self.lr_decay(episode, self.q_value_lr)
		# for param_group in self.q_critic_optimizer.param_groups:
		# 	param_group['lr'] = q_value_lr

		# policy_lr = self.lr_decay(episode, self.policy_lr)
		# for param_group in self.policy_optimizer.param_groups:
		# 	param_group['lr'] = policy_lr

		print(v_value_lr, q_value_lr, policy_lr)

		with torch.no_grad():
			# OLD VALUES
			Q_values_old, weights_prd_old, _, h_q = self.critic_network_q(
												old_states_critic_allies.to(self.device),
												old_states_critic_enemies.to(self.device),
												old_one_hot_actions.to(self.device),
												rnn_hidden_state_q.to(self.device)
												)
			Values_old, _, _, h_v = self.critic_network_v(
												old_states_critic_allies.to(self.device),
												old_states_critic_enemies.to(self.device),
												old_one_hot_actions.to(self.device),
												rnn_hidden_state_v.to(self.device)
												)

			next_Values_old, _, _, _ = self.critic_network_v(
										torch.FloatTensor(np.array(self.buffer.states_critic_allies))[:, -1, :, :].to(self.device).unsqueeze(1),
										torch.FloatTensor(np.array(self.buffer.states_critic_enemies))[:, -1, :, :].to(self.device).unsqueeze(1),
										torch.FloatTensor(np.array(self.buffer.one_hot_actions))[:, -1, :].to(self.device).unsqueeze(1),
										h_v.to(self.device)
										)

			next_Q_values_old, _, _, _ = self.critic_network_q(
										torch.FloatTensor(np.array(self.buffer.states_critic_allies))[:, -1, :, :].to(self.device).unsqueeze(1),
										torch.FloatTensor(np.array(self.buffer.states_critic_enemies))[:, -1, :, :].to(self.device).unsqueeze(1),
										torch.FloatTensor(np.array(self.buffer.one_hot_actions))[:, -1, :].to(self.device).unsqueeze(1),
										h_q.to(self.device)
										)

			next_mask = 1-torch.FloatTensor(np.array(self.buffer.dones))[:, -1, :].long().reshape(-1, self.num_agents)


		# if "prd_above_threshold_ascend" in self.experiment_type or "prd_above_threshold_decay" in self.experiment_type:
		# 	mask_rewards = (torch.mean(weights_prd_old, dim=1)>self.select_above_threshold).int()
		# 	target_V_rewards = torch.sum(rewards.unsqueeze(-2).repeat(1, self.num_agents, 1) * torch.transpose(mask_rewards.cpu(),-1,-2), dim=-1)
		# elif "threshold" in self.experiment_type and episode > self.steps_to_take:
		# 	mask_rewards = (torch.mean(weights_prd_old, dim=1)>self.select_above_threshold).int()
		# 	target_V_rewards = torch.sum(rewards.unsqueeze(-2).repeat(1, self.num_agents, 1) * torch.transpose(mask_rewards.cpu(),-1,-2), dim=-1)
		# elif "top" in self.experiment_type and episode > self.steps_to_take:
		# 	values, indices = torch.topk(torch.mean(weights_prd_old, dim=1), k=self.top_k, dim=-1)
		# 	mask_rewards = torch.sum(F.one_hot(indices, num_classes=self.num_agents), dim=-2)
		# 	target_V_rewards = torch.sum(rewards.unsqueeze(-2).repeat(1, self.num_agents, 1) * torch.transpose(mask_rewards.cpu(),-1,-2), dim=-1)
		# elif "prd_soft_advantage" in self.experiment_type and episode > self.steps_to_take:
		# 	target_V_rewards = torch.sum(rewards.unsqueeze(-2).repeat(1, self.num_agents, 1) * torch.transpose(torch.mean(weights_prd_old.cpu(), dim=1), -1, -2), dim=-1)
		# else:
		# 	target_V_rewards = torch.sum(rewards.unsqueeze(-2).repeat(1, self.num_agents, 1), dim=-1)

		# shape = (batch, time_steps, self.num_agents)
		# target_Qs = rewards.reshape(*shape).to(self.device) + self.gamma * (1-dones.reshape(batch, time_steps+1, self.num_agents).to(self.device)[:,1:,:]) * Q_values_old.reshape(batch, time_steps+1, self.num_agents)[:,1:,:]
		# target_Vs = target_V_rewards.reshape(*shape).to(self.device) + self.gamma * (1-dones.reshape(batch, time_steps+1, self.num_agents).to(self.device)[:,1:,:]) * Values_old.reshape(batch, time_steps+1, self.num_agents)[:,1:,:]
		# target_Q_values = self.build_td_lambda_targets(rewards.reshape(*shape).to(self.device), dones.reshape(batch, time_steps+1, self.num_agents).to(self.device), masks.reshape(*shape[:-1]).to(self.device), Q_values_old.reshape(batch, time_steps+1, self.num_agents)[:,:-1,:]).reshape(-1, self.num_agents)
		# target_V_values = self.build_td_lambda_targets(target_V_rewards.reshape(*shape).to(self.device), dones.reshape(batch, time_steps+1, self.num_agents).to(self.device), masks.reshape(*shape[:-1]).to(self.device), Values_old.reshape(batch, time_steps+1, self.num_agents)[:,:-1,:]).reshape(-1, self.num_agents)

		if self.norm_returns:
			values_shape = Values_old.shape
			Values_old_ = self.v_value_norm.denormalize(Values_old.view(-1)).view(values_shape)*masks.view(values_shape).to(self.device)
			# Values_old_ = (Values_old * torch.sqrt(self.v_value_norm.var) + self.v_value_norm.mean)*masks.view(values_shape).to(self.device)
			values_shape = Q_values_old.shape
			Q_values_old_ = self.q_value_norm.denormalize(Q_values_old.view(-1)).view(values_shape)*masks.view(values_shape).to(self.device)
			# Q_values_old_ = (Values_old * torch.sqrt(self.q_value_norm.var) + self.q_value_norm.mean)*masks.view(values_shape).to(self.device)

			values_shape = next_Values_old.shape
			next_Values_old = self.v_value_norm.denormalize(next_Values_old.view(-1)).view(values_shape)*next_mask.view(values_shape).to(self.device)
			values_shape = next_Q_values_old.shape
			next_Q_values_old = self.q_value_norm.denormalize(next_Q_values_old.view(-1)).view(values_shape)*next_mask.view(values_shape).to(self.device)
			# next_Values_old = (next_Values_old * torch.sqrt(self.v_value_norm.var) + self.v_value_norm.mean)*next_mask.to(self.device)
			# next_Q_values_old = (next_Q_values_old * torch.sqrt(self.q_value_norm.var) + self.q_value_norm.mean)*next_mask.to(self.device)

			advantage, masking_rewards, mean_min_weight_value = self.calculate_advantages_based_on_exp(Values_old_, next_Values_old, rewards.to(self.device), dones.to(self.device), torch.mean(weights_prd_old, dim=1), masks.to(self.device), next_mask.to(self.device), episode)
			target_V_values = (Values_old_ + advantage)*masks.reshape(-1, self.num_agents).to(self.device) # gae return
			advantage_Q = self.calculate_advantages(Q_values_old_, next_Q_values_old, rewards.to(self.device), dones.to(self.device), masks.to(self.device), next_mask.to(self.device))
			target_Q_values = (Q_values_old_ + advantage_Q)*masks.reshape(-1, self.num_agents).to(self.device)

		else:
			advantage, masking_rewards, mean_min_weight_value = self.calculate_advantages_based_on_exp(Values_old, next_Values_old, rewards.to(self.device), dones.to(self.device), torch.mean(weights_prd_old, dim=1), masks.to(self.device), next_mask.to(self.device), episode)
			target_V_values = (Values_old + advantage)*masks.reshape(-1, self.num_agents).to(self.device) # gae return
			advantage_Q = self.calculate_advantages(Q_values_old, next_Q_values_old, rewards.to(self.device), dones.to(self.device), masks.to(self.device), next_mask.to(self.device))
			target_Q_values = (Q_values_old + advantage_Q)*masks.reshape(-1, self.num_agents).to(self.device)

		# print("target_V_values")
		# print(target_V_values[0])
		# print("advantages")
		# print(advantage[0])

		if self.norm_returns:
			targets_shape = target_V_values.shape
			# targets = targets.reshape(-1)
			self.v_value_norm.update(target_V_values.view(-1), masks.view(-1).to(self.device))
			target_V_values = self.v_value_norm.normalize(target_V_values.view(-1)).view(targets_shape) * masks.view(targets_shape).to(self.device)

			targets_shape = target_Q_values.shape
			# targets = targets.reshape(-1)
			self.q_value_norm.update(target_Q_values.view(-1), masks.view(-1).to(self.device))
			target_Q_values = self.q_value_norm.normalize(target_Q_values.view(-1)).view(targets_shape) * masks.view(targets_shape).to(self.device)

			# self.v_value_norm.update(target_V_values.view(-1, self.num_agents), masks.view(-1, self.num_agents).to(self.device))
			# target_V_values = (target_V_values - self.v_value_norm.mean) / (torch.sqrt(self.v_value_norm.var) + 1e-5)
			# self.q_value_norm.update(target_Q_values.view(-1, self.num_agents), masks.view(-1, self.num_agents).to(self.device))
			# target_Q_values = (target_Q_values - self.q_value_norm.mean) / (torch.sqrt(self.q_value_norm.var) + 1e-5)

		# if self.norm_adv:
		# 	# advantage = advantage.reshape(self.update_ppo_agent, -1, self.num_agents)
		# 	# advantage = ((advantage - advantage.mean(dim=1).unsqueeze(1)) / advantage.std(dim=1).unsqueeze(1))*masks.to(self.device)
		# 	# advantage = advantage.reshape(-1, self.num_agents)
		# 	advantage_mean = (advantage*masks.view(-1, self.num_agents).to(self.device)).sum()/masks.sum()
		# 	advantage_std = ((((advantage-advantage_mean)*masks.view(-1, self.num_agents).to(self.device))**2).sum()/masks.sum())**0.5
		# 	advantage = ((advantage - advantage_mean) / (advantage_std + 1e-6))*masks.view(-1, self.num_agents).to(self.device)

		# print("target_qs")
		# print(target_Q_values[0])
		# print(target_Q_values[-1])
		# print("target_vs")
		# print(target_V_values[0])
		# print(target_V_values[-1])

		# if self.norm_returns:
		# 	target_Q_values = (target_Q_values - target_Q_values.mean()) / target_Q_values.std()
		# 	target_V_values = (target_V_values - target_V_values.mean()) / target_V_values.std()

		# print("target_qs")
		# print(target_Q_values[0])
		# print(target_Q_values[-1])
		# print("target_vs")
		# print(target_V_values[0])
		# print(target_V_values[-1])

		rewards = rewards.reshape(-1, self.num_agents)
		dones = dones.reshape(-1, self.num_agents)
		# masks = masks.reshape(-1, 1)
		masks = masks.reshape(-1, self.num_agents)

		q_value_loss_batch = 0
		v_value_loss_batch = 0
		policy_loss_batch = 0
		entropy_batch = 0
		weight_prd_batch = None
		weight_v_batch = None
		grad_norm_value_v_batch = 0
		grad_norm_value_q_batch = 0
		grad_norm_policy_batch = 0
		agent_groups_over_episode_batch = 0
		avg_agent_group_over_episode_batch = 0

		
		# torch.autograd.set_detect_anomaly(True)
		# Optimize policy for n epochs
		for _ in range(self.n_epochs):

			Q_values, weights_prd, score_q, _ = self.critic_network_q(
												old_states_critic_allies.to(self.device),
												old_states_critic_enemies.to(self.device),
												old_one_hot_actions.to(self.device),
												rnn_hidden_state_q.to(self.device)
												)
			Values, weight_v, score_v, h_v = self.critic_network_v(
												old_states_critic_allies.to(self.device),
												old_states_critic_enemies.to(self.device),
												old_one_hot_actions.to(self.device),
												rnn_hidden_state_v.to(self.device)
												)


			dists, _ = self.policy_network(
					torch.cat([old_states_actor, old_last_one_hot_actions], dim=-1).to(self.device),
					rnn_hidden_state_actor.to(self.device),
					old_mask_actions.to(self.device),
					update=True
					)

			# weights_prd = weights_prd.reshape(batch, time_steps+1, -1, self.num_agents, self.num_agents)[:,:-1,:,:,:].reshape(batch*time_steps, -1, self.num_agents, self.num_agents)
			# weight_v = weight_v.reshape(batch, time_steps+1, -1, self.num_agents, self.num_agents)[:,:-1,:,:,:].reshape(batch*time_steps, -1, self.num_agents, self.num_agents)

			next_Values, _, _, _ = self.critic_network_v(
										torch.FloatTensor(np.array(self.buffer.states_critic_allies))[:, -1, :, :].to(self.device).unsqueeze(1),
										torch.FloatTensor(np.array(self.buffer.states_critic_enemies))[:, -1, :, :].to(self.device).unsqueeze(1),
										torch.FloatTensor(np.array(self.buffer.one_hot_actions))[:, -1, :].to(self.device).unsqueeze(1),
										h_v.to(self.device)
										)

			if self.norm_returns:
				targets_shape = next_Values.shape
				next_Values = self.v_value_norm.denormalize(next_Values.view(-1)).view(targets_shape)*next_mask.to(self.device)
				targets_shape = Values.shape
				Values = self.v_value_norm.denormalize(Values.view(-1)).view(targets_shape)*masks.to(self.device)
			
			advantage, masking_rewards, mean_min_weight_value = self.calculate_advantages_based_on_exp(Values, next_Values, rewards.to(self.device), dones.to(self.device), torch.mean(weights_prd, dim=1), masks.to(self.device), next_mask.to(self.device), episode)
			# print("advantage")
			# print(advantage[0])
			# advantage, masking_rewards, mean_min_weight_value = self.calculate_advantages_based_on_exp(Values, Values, rewards.to(self.device), dones.to(self.device), torch.mean(weights_prd.detach(), dim=1), masks.to(self.device), episode)
			if self.norm_returns:
				targets_shape = Values.shape
				Values = self.v_value_norm.normalize(Values.view(-1)).view(targets_shape)
				targets_shape = next_Values.shape
				next_Values = self.v_value_norm.normalize(next_Values.view(-1)).view(targets_shape)

			if self.norm_adv:
				advantage_mean = (advantage*masks.view(-1, self.num_agents).to(self.device)).sum()/masks.sum()
				advantage_std = ((((advantage-advantage_mean)*masks.view(-1, self.num_agents).to(self.device))**2).sum()/masks.sum())**0.5
				advantage = ((advantage - advantage_mean) / (advantage_std + 1e-6))*masks.view(-1, self.num_agents).to(self.device)

			# print(advantage[0])
			# print(advantage[-1])

			probs = Categorical(dists)
			logprobs = probs.log_prob(old_actions.to(self.device) * masks.to(self.device))
			# print("categorical logprobs")
			# print(logprobs[0])
			
			# dists_taken = torch.gather(dists, dim=-1, index=old_actions.unsqueeze(-1).long().to(self.device)).squeeze(-1)
			# logprobs = torch.log(dists_taken + 1e-10) * masks.to(self.device)
			# print("logprobs")
			# print(logprobs[0])
			
			if "threshold" in self.experiment_type or "top" in self.experiment_type:
				agent_groups_over_episode = torch.sum(torch.sum(masking_rewards.reshape(-1, self.num_agents, self.num_agents).float(), dim=-2),dim=0)/masking_rewards.reshape(-1, self.num_agents, self.num_agents).shape[0]
				avg_agent_group_over_episode = torch.mean(agent_groups_over_episode)
				agent_groups_over_episode_batch += agent_groups_over_episode
				avg_agent_group_over_episode_batch += avg_agent_group_over_episode
				

			critic_v_loss_1 = F.huber_loss(Values*masks.to(self.device), target_V_values*masks.to(self.device), reduction="sum", delta=10.0) / masks.sum() #(self.num_agents*masks.sum())
			critic_v_loss_2 = F.huber_loss(torch.clamp(Values, Values_old.to(self.device)-self.value_clip, Values_old.to(self.device)+self.value_clip)*masks.to(self.device), target_V_values*masks.to(self.device), reduction="sum", delta=10.0) / masks.sum() #(self.num_agents*masks.sum()
			# critic_v_loss = F.huber_loss(Values*masks.to(self.device), target_V_values*masks.to(self.device), reduction="sum", delta=10.0) / masks.sum() #(self.num_agents*masks.sum())

			critic_q_loss_1 = F.huber_loss(Q_values*masks.to(self.device), target_Q_values*masks.to(self.device), reduction="sum", delta=10.0) / masks.sum() #(self.num_agents*masks.sum())
			critic_q_loss_2 = F.huber_loss(torch.clamp(Q_values, Q_values_old.to(self.device)-self.value_clip, Q_values_old.to(self.device)+self.value_clip)*masks.to(self.device), target_Q_values*masks.to(self.device), reduction="sum", delta=10.0) / masks.sum() #(self.num_agents*masks.sum())
			# critic_q_loss = F.huber_loss(Q_values*masks.to(self.device), target_Q_values*masks.to(self.device), reduction="sum", delta=10.0) / masks.sum() #(self.num_agents*masks.sum())

			entropy_weights = 0
			entropy_weights_v = 0
			score_q_cum = 0
			score_v_cum = 0
			for i in range(self.num_heads):
				# entropy_weights += -torch.mean(torch.sum((weights_prd[:, i] * torch.log(torch.clamp(weights_prd[:, i], 1e-10, 1.0)) * masks.unsqueeze(-1).to(self.device)), dim=-1))
				# entropy_weights_v += -torch.mean(torch.sum(weight_v[:, i] * torch.log(torch.clamp(weight_v[:, i], 1e-10, 1.0)) * masks.unsqueeze(-1).to(self.device), dim=-1))
				entropy_weights += -torch.sum(torch.sum((weights_prd[:, i] * torch.log(torch.clamp(weights_prd[:, i], 1e-10, 1.0)) * masks.unsqueeze(-1).to(self.device)), dim=-1))/masks.sum() #(masks.sum()*self.num_agents)
				entropy_weights_v += -torch.sum(torch.sum(weight_v[:, i] * torch.log(torch.clamp(weight_v[:, i], 1e-10, 1.0)) * masks.unsqueeze(-1).to(self.device), dim=-1))/masks.sum() #(masks.sum()*self.num_agents)
				
				score_q_cum += (score_q[:, i].squeeze(-2)**2 * masks.unsqueeze(-1).to(self.device)).sum()/masks.sum()
				score_v_cum += (score_v[:, i].squeeze(-2)**2 * masks.unsqueeze(-1).to(self.device)).sum()/masks.sum()
			
			# print("weights entropy")
			# print(entropy_weights.item()/weights_prd.shape[1], entropy_weights_v.item()/weight_v.shape[1])

			critic_q_loss = torch.max(critic_q_loss_1, critic_q_loss_2) + self.critic_score_regularizer*score_q_cum + self.critic_weight_entropy_pen*entropy_weights
			critic_v_loss = torch.max(critic_v_loss_1, critic_v_loss_2) + self.critic_score_regularizer*score_v_cum + self.critic_weight_entropy_pen*entropy_weights_v


			# Finding the ratio (pi_theta / pi_theta__old)
			ratios = torch.exp((logprobs - old_logprobs.to(self.device))*masks.to(self.device))
			# print("*"*20)
			# print(ratios[12])
			# Finding Surrogate Loss
			surr1 = ratios * advantage * masks.to(self.device)
			surr2 = torch.clamp(ratios, 1-self.policy_clip, 1+self.policy_clip) * advantage * masks.to(self.device)

			# final loss of clipped objective PPO
			# entropy = -torch.mean(torch.sum(dists*masks.unsqueeze(-1).to(self.device) * torch.log(torch.clamp(dists*masks.unsqueeze(-1).to(self.device), 1e-10,1.0)), dim=-1))
			entropy = -torch.sum(torch.sum(dists*masks.unsqueeze(-1).to(self.device) * torch.log(torch.clamp(dists*masks.unsqueeze(-1).to(self.device), 1e-10,1.0)), dim=-1))/ masks.sum() #(masks.sum()*self.num_agents)
			# policy_loss = (-torch.min(surr1, surr2).mean() - self.entropy_pen*entropy)
			# policy_loss = ((-torch.min(surr1, surr2).sum())/(masks.sum()*self.num_agents) - self.entropy_pen*entropy)
			policy_loss_ = (-torch.min(surr1, surr2).sum())/masks.sum()
			policy_loss = policy_loss_ - self.entropy_pen*entropy

			print("loss", "*"*20)
			print(policy_loss_.item(), policy_loss.item(), entropy.item(), self.entropy_pen*entropy.item(), critic_v_loss.item(), critic_q_loss.item())

			self.q_critic_optimizer.zero_grad()
			critic_q_loss.backward()
			if self.enable_grad_clip_critic:
				grad_norm_value_q = torch.nn.utils.clip_grad_norm_(self.critic_network_q.parameters(), self.grad_clip_critic)
			else:
				total_norm = 0
				for p in self.critic_network_q.parameters():
					param_norm = p.grad.detach().data.norm(2)
					total_norm += param_norm.item() ** 2
				grad_norm_value_q = torch.tensor([total_norm ** 0.5])
			self.q_critic_optimizer.step()

			# self.critic_network_q.rnn_hidden_state = None #rnn_hidden_state_q.detach()
			
			self.v_critic_optimizer.zero_grad()
			critic_v_loss.backward()
			if self.enable_grad_clip_critic:
				grad_norm_value_v = torch.nn.utils.clip_grad_norm_(self.critic_network_v.parameters(), self.grad_clip_critic)
			else:
				total_norm = 0
				for p in self.critic_network_v.parameters():
					param_norm = p.grad.detach().data.norm(2)
					total_norm += param_norm.item() ** 2
				grad_norm_value_v = torch.tensor([total_norm ** 0.5])
			self.v_critic_optimizer.step()

			# self.critic_network_v.rnn_hidden_state = None #rnn_hidden_state_v.detach()
			

			self.policy_optimizer.zero_grad()
			policy_loss.backward()
			if self.enable_grad_clip_actor:
				grad_norm_policy = torch.nn.utils.clip_grad_norm_(self.policy_network.parameters(), self.grad_clip_actor)
			else:
				total_norm = 0
				for p in self.policy_network.parameters():
					param_norm = p.grad.detach().data.norm(2)
					total_norm += param_norm.item() ** 2
				grad_norm_policy = torch.tensor([total_norm ** 0.5])
			self.policy_optimizer.step()

			# print("grads")
			# print(grad_norm_value_q.item(), grad_norm_value_v.item(), grad_norm_policy.item())

			# self.policy_network.rnn_hidden_state = None #rnn_hidden_state_actor.detach()

			q_value_loss_batch += critic_q_loss.item()
			v_value_loss_batch += critic_v_loss.item()
			policy_loss_batch += policy_loss.item()
			entropy_batch += entropy.item()
			grad_norm_value_v_batch += grad_norm_value_v
			grad_norm_value_q_batch += grad_norm_value_q
			grad_norm_policy_batch += grad_norm_policy
			if weight_prd_batch is None:
				weight_prd_batch = weights_prd.detach().cpu()
			else:
				weight_prd_batch += weights_prd.detach().cpu()
			if weight_v_batch is None:
				weight_v_batch = weight_v.detach().cpu()
			else:
				weight_v_batch += weight_v.detach().cpu()
			


		# if episode % self.network_update_interval == 0:
		# 	# Copy new weights into old critic
		# 	self.critic_network_q_old.load_state_dict(self.critic_network_q.state_dict())
		# 	self.critic_network_v_old.load_state_dict(self.critic_network_v.state_dict())

		# 	# Copy new weights into old policy
		# 	self.policy_network_old.load_state_dict(self.policy_network.state_dict())

		# self.scheduler.step()
		# print("learning rate of policy", self.scheduler.get_lr())

		# clear buffer
		self.buffer.clear()
		

		q_value_loss_batch /= self.n_epochs
		v_value_loss_batch /= self.n_epochs
		policy_loss_batch /= self.n_epochs
		entropy_batch /= self.n_epochs
		grad_norm_value_v_batch /= self.n_epochs
		grad_norm_value_q_batch /= self.n_epochs
		grad_norm_policy_batch /= self.n_epochs
		weight_prd_batch /= self.n_epochs
		weight_v_batch /= self.n_epochs
		agent_groups_over_episode_batch /= self.n_epochs
		avg_agent_group_over_episode_batch /= self.n_epochs


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
		"grad_norm_value_v":grad_norm_value_v_batch,
		"grad_norm_value_q": grad_norm_value_q_batch,
		"grad_norm_policy": grad_norm_policy_batch,
		"weights_prd": weight_prd_batch,
		"weights_v": weight_v_batch,
		"v_value_lr": v_value_lr,
		"q_value_lr": q_value_lr,
		"policy_lr": policy_lr,
		}

		if "threshold" in self.experiment_type:
			self.plotting_dict["agent_groups_over_episode"] = agent_groups_over_episode_batch
			self.plotting_dict["avg_agent_group_over_episode"] = avg_agent_group_over_episode_batch
		if "prd_top" in self.experiment_type:
			self.plotting_dict["mean_min_weight_value"] = mean_min_weight_value

		if self.comet_ml is not None:
			self.plot(masks, episode)

		del q_value_loss_batch, v_value_loss_batch, policy_loss_batch, entropy_batch, grad_norm_value_v_batch, grad_norm_value_q_batch, grad_norm_policy_batch, weight_prd_batch, agent_groups_over_episode_batch, avg_agent_group_over_episode_batch
		torch.cuda.empty_cache()
