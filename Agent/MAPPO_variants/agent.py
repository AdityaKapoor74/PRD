import numpy as np
import math
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from model import Policy, Global_Q_network, Q_network, V_network
from utils import RolloutBuffer

class PPOAgent:

	def __init__(
		self, 
		env, 
		dictionary,
		comet_ml,
		):

		# Environment Setup
		self.environment = dictionary["environment"]
		self.env = env
		self.env_name = dictionary["env"]
		self.num_agents = dictionary["num_agents"]
		self.num_actions = dictionary["num_actions"]

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
		self.rnn_num_layers_q = dictionary["rnn_num_layers_q"]
		self.rnn_num_layers_v = dictionary["rnn_num_layers_v"]
		self.rnn_hidden_q = dictionary["rnn_hidden_q"]
		self.rnn_hidden_v = dictionary["rnn_hidden_v"]
		self.critic_ally_observation = dictionary["ally_observation"]

		if "StarCraft" in self.environment:
			self.num_enemies = self.env.n_enemies
			self.critic_enemy_observation = dictionary["enemy_observation"]
		else:
			self.num_enemies = 1
			self.critic_enemy_observation = 1

		self.q_value_lr = dictionary["q_value_lr"]
		self.v_value_lr = dictionary["v_value_lr"]
		self.q_weight_decay = dictionary["q_weight_decay"]
		self.v_weight_decay = dictionary["v_weight_decay"]
		self.critic_weight_entropy_pen = dictionary["critic_weight_entropy_pen"]
		self.critic_weight_entropy_pen_final = dictionary["critic_weight_entropy_pen_final"]
		self.critic_weight_entropy_pen_decay_rate = (dictionary["critic_weight_entropy_pen_final"] - dictionary["critic_weight_entropy_pen"]) / dictionary["critic_weight_entropy_pen_steps"]
		self.critic_score_regularizer = dictionary["critic_score_regularizer"]
		self.target_calc_style = dictionary["target_calc_style"]
		self.n_steps = dictionary["n_steps"]
		self.value_clip = dictionary["value_clip"]
		self.num_heads = dictionary["num_heads"]
		self.enable_hard_attention = dictionary["enable_hard_attention"]
		self.enable_grad_clip_critic_v = dictionary["enable_grad_clip_critic_v"]
		self.grad_clip_critic_v = dictionary["grad_clip_critic_v"]
		self.enable_grad_clip_critic_q = dictionary["enable_grad_clip_critic_q"]
		self.grad_clip_critic_q = dictionary["grad_clip_critic_q"]
		self.norm_returns_q = dictionary["norm_returns_q"]
		self.norm_returns_v = dictionary["norm_returns_v"]
		self.clamp_rewards = dictionary["clamp_rewards"]
		self.clamp_rewards_value_min = dictionary["clamp_rewards_value_min"]
		self.clamp_rewards_value_max = dictionary["clamp_rewards_value_max"]


		# Actor Setup
		self.data_chunk_length = dictionary["data_chunk_length"]
		self.warm_up = dictionary["warm_up"]
		self.warm_up_episodes = dictionary["warm_up_episodes"]
		self.epsilon_start = self.epsilon = dictionary["epsilon_start"]
		self.epsilon_end = dictionary["epsilon_end"]
		self.rnn_num_layers_actor = dictionary["rnn_num_layers_actor"]
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
		Q_PopArt = None
		if "prd" in self.experiment_type:

			if self.experiment_type == "prd_soft_advantage_global":

				self.critic_network_q = Global_Q_network(
					ally_obs_input_dim=self.critic_ally_observation, 
					enemy_obs_input_dim=self.critic_enemy_observation, 
					num_heads=self.num_heads, 
					num_agents=self.num_agents, 
					num_enemies=self.num_enemies, 
					num_actions=self.num_actions, 
					rnn_num_layers=self.rnn_num_layers_q,
					device=self.device, 
					enable_hard_attention=self.enable_hard_attention, 
					attention_dropout_prob=dictionary["attention_dropout_prob_q"], 
					temperature=self.temperature_q,
					norm_returns=self.norm_returns_q,
					environment=self.environment,
					).to(self.device)
				self.target_critic_network_q = Global_Q_network(
					ally_obs_input_dim=self.critic_ally_observation, 
					enemy_obs_input_dim=self.critic_enemy_observation, 
					num_heads=self.num_heads, 
					num_agents=self.num_agents, 
					num_enemies=self.num_enemies, 
					num_actions=self.num_actions, 
					rnn_num_layers=self.rnn_num_layers_q,
					device=self.device, 
					enable_hard_attention=self.enable_hard_attention, 
					attention_dropout_prob=dictionary["attention_dropout_prob_q"], 
					temperature=self.temperature_q,
					norm_returns=self.norm_returns_q,
					environment=self.environment,
					).to(self.device)

			else:

				self.critic_network_q = Q_network(
					ally_obs_input_dim=self.critic_ally_observation, 
					enemy_obs_input_dim=self.critic_enemy_observation, 
					num_heads=self.num_heads, 
					num_agents=self.num_agents, 
					num_enemies=self.num_enemies, 
					num_actions=self.num_actions, 
					rnn_num_layers=self.rnn_num_layers_q,
					device=self.device, 
					enable_hard_attention=self.enable_hard_attention, 
					attention_dropout_prob=dictionary["attention_dropout_prob_q"], 
					temperature=self.temperature_q,
					norm_returns=self.norm_returns_q,
					environment=self.environment,
					).to(self.device)
				self.target_critic_network_q = Q_network(
					ally_obs_input_dim=self.critic_ally_observation, 
					enemy_obs_input_dim=self.critic_enemy_observation, 
					num_heads=self.num_heads, 
					num_agents=self.num_agents, 
					num_enemies=self.num_enemies, 
					num_actions=self.num_actions, 
					rnn_num_layers=self.rnn_num_layers_q,
					device=self.device, 
					enable_hard_attention=self.enable_hard_attention, 
					attention_dropout_prob=dictionary["attention_dropout_prob_q"], 
					temperature=self.temperature_q,
					norm_returns=self.norm_returns_q,
					environment=self.environment,
					).to(self.device)

			# Copy network params
			self.target_critic_network_q.load_state_dict(self.critic_network_q.state_dict())
			# Disable updates for old network
			for param in self.target_critic_network_q.parameters():
				param.requires_grad_(False)

			Q_PopArt = self.critic_network_q.q_value_layer[-1]

		
		self.critic_network_v = V_network(
			ally_obs_input_dim=self.critic_ally_observation, 
			enemy_obs_input_dim=self.critic_enemy_observation, 
			num_heads=self.num_heads, 
			num_agents=self.num_agents, 
			num_enemies=self.num_enemies, 
			num_actions=self.num_actions, 
			rnn_num_layers=self.rnn_num_layers_v,
			device=self.device, 
			enable_hard_attention=self.enable_hard_attention, 
			attention_dropout_prob=dictionary["attention_dropout_prob_v"], 
			temperature=self.temperature_v,
			norm_returns=self.norm_returns_v,
			environment=self.environment,
			).to(self.device)
		self.target_critic_network_v = V_network(
			ally_obs_input_dim=self.critic_ally_observation, 
			enemy_obs_input_dim=self.critic_enemy_observation, 
			num_heads=self.num_heads, 
			num_agents=self.num_agents, 
			num_enemies=self.num_enemies, 
			num_actions=self.num_actions,
			rnn_num_layers=self.rnn_num_layers_v, 
			device=self.device, 
			enable_hard_attention=self.enable_hard_attention, 
			attention_dropout_prob=dictionary["attention_dropout_prob_v"], 
			temperature=self.temperature_v,
			norm_returns=self.norm_returns_v,
			environment=self.environment,
			).to(self.device)
		# Copy network params
		self.target_critic_network_v.load_state_dict(self.critic_network_v.state_dict())
		# Disable updates for old network
		for param in self.target_critic_network_v.parameters():
			param.requires_grad_(False)

		V_PopArt = self.critic_network_v.v_value_layer[-1]
		
		
		# Policy Network
		if self.experiment_type == "HAPPO":
			self.policy_network = []
			for i in range(self.num_agents):
				self.policy_network.append(Policy(
					obs_input_dim=self.actor_observation_shape, 
					num_agents=self.num_agents, 
					num_actions=self.num_actions, 
					rnn_num_layers=self.rnn_num_layers_actor,
					device=self.device
					).to(self.device))
		else:
			self.policy_network = Policy(
				obs_input_dim=self.actor_observation_shape, 
				num_agents=self.num_agents, 
				num_actions=self.num_actions, 
				rnn_num_layers=self.rnn_num_layers_actor,
				device=self.device
				).to(self.device)
		

		self.network_update_interval_q = dictionary["network_update_interval_q"]
		self.network_update_interval_v = dictionary["network_update_interval_v"]
		self.soft_update_q = dictionary["soft_update_q"]
		self.soft_update_v = dictionary["soft_update_v"]
		self.tau_q = dictionary["tau_q"]
		self.tau_v = dictionary["tau_v"]


		self.buffer = RolloutBuffer(
			environment=self.environment,
			experiment_type=self.experiment_type,
			num_episodes=self.update_ppo_agent, 
			max_time_steps=self.max_time_steps, 
			num_agents=self.num_agents, 
			num_enemies=self.num_enemies,
			obs_shape_critic_ally=self.critic_ally_observation, 
			obs_shape_critic_enemy=self.critic_enemy_observation, 
			obs_shape_actor=self.actor_observation_shape, 
			rnn_num_layers_actor=self.rnn_num_layers_actor,
			actor_hidden_state=self.rnn_hidden_actor,
			rnn_num_layers_q=self.rnn_num_layers_q,
			rnn_num_layers_v=self.rnn_num_layers_v,
			q_hidden_state=self.rnn_hidden_q,
			v_hidden_state=self.rnn_hidden_v,
			num_actions=self.num_actions,
			transition_after=self.steps_to_take,
			data_chunk_length=self.data_chunk_length,
			norm_returns_q=self.norm_returns_q,
			norm_returns_v=self.norm_returns_v,
			clamp_rewards=self.clamp_rewards,
			clamp_rewards_value_min=self.clamp_rewards_value_min,
			clamp_rewards_value_max=self.clamp_rewards_value_max,
			norm_rewards=self.norm_rewards,
			target_calc_style=self.target_calc_style,
			gae_lambda=self.gae_lambda,
			n_steps=self.n_steps,
			gamma=self.gamma,
			V_PopArt=V_PopArt,
			Q_PopArt=Q_PopArt,
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


		self.v_critic_optimizer = optim.AdamW(self.critic_network_v.parameters(), lr=self.v_value_lr, weight_decay=self.v_weight_decay, eps=1e-05)
		
		if self.experiment_type == "HAPPO":
			self.policy_optimizer = []
			for agent_id in range(self.num_agents):
				self.policy_optimizer.append(optim.AdamW(self.policy_network[agent_id].parameters(),lr=self.policy_lr, weight_decay=self.policy_weight_decay, eps=1e-05))
		else:
			self.policy_optimizer = optim.AdamW(self.policy_network.parameters(),lr=self.policy_lr, weight_decay=self.policy_weight_decay, eps=1e-05)

		if "prd" in self.experiment_type:
			self.q_critic_optimizer = optim.AdamW(self.critic_network_q.parameters(), lr=self.q_value_lr, weight_decay=self.q_weight_decay, eps=1e-05)

		if self.scheduler_need:
			if self.experiment_type == "HAPPO":
				self.scheduler_policy = [optim.lr_scheduler.MultiStepLR(self.policy_optimizer, milestones=[1000, 20000], gamma=0.1) for i in range(self.num_agents)]
			else:
				self.scheduler_policy = optim.lr_scheduler.MultiStepLR(self.policy_optimizer, milestones=[1000, 20000], gamma=0.1)
			self.scheduler_v_critic = optim.lr_scheduler.MultiStepLR(self.v_critic_optimizer, milestones=[1000, 20000], gamma=0.1)
			if "prd" in self.experiment_type:
				self.scheduler_q_critic = optim.lr_scheduler.MultiStepLR(self.q_critic_optimizer, milestones=[1000, 20000], gamma=0.1)

		self.comet_ml = None
		if dictionary["save_comet_ml_plot"]:
			self.comet_ml = comet_ml

	
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

	
	def get_q_v_values(self, state_allies, state_enemies, one_hot_actions, rnn_hidden_state_q, rnn_hidden_state_v, indiv_dones):
		with torch.no_grad():
			indiv_masks = [1-d for d in indiv_dones]
			indiv_masks = torch.FloatTensor(indiv_masks).unsqueeze(0).unsqueeze(0)
			state_allies = torch.FloatTensor(state_allies).unsqueeze(0).unsqueeze(0)
			if "StarCraft" in self.environment:
				state_enemies = torch.FloatTensor(state_enemies).unsqueeze(0).unsqueeze(0)
			one_hot_actions = torch.FloatTensor(one_hot_actions).unsqueeze(0).unsqueeze(0)
			rnn_hidden_state_v = torch.FloatTensor(rnn_hidden_state_v)
			if "prd" in self.experiment_type:
				rnn_hidden_state_q = torch.FloatTensor(rnn_hidden_state_q)
				

			if self.experiment_type == "prd_soft_advantage_global":
				if "StarCraft" in self.environment:
					Value, _, _, rnn_hidden_state_v = self.target_critic_network_v(state_allies.to(self.device), state_enemies.to(self.device), one_hot_actions.to(self.device), rnn_hidden_state_v.to(self.device), indiv_masks.to(self.device))
					Q_value, global_weights, weights_prd, _, score, rnn_hidden_state_q = self.target_critic_network_q(state_allies.to(self.device), state_enemies.to(self.device), one_hot_actions.to(self.device), rnn_hidden_state_q.to(self.device), indiv_masks.to(self.device))
				else:
					Value, _, _, rnn_hidden_state_v = self.target_critic_network_v(state_allies.to(self.device), None, one_hot_actions.to(self.device), rnn_hidden_state_v.to(self.device), indiv_masks.to(self.device))
					Q_value, global_weights, weights_prd, _, score, rnn_hidden_state_q = self.target_critic_network_q(state_allies.to(self.device), None, one_hot_actions.to(self.device), rnn_hidden_state_q.to(self.device), indiv_masks.to(self.device))

				# we repeath Q value because we assume that a global Q has been computed for each agent
				return Q_value.reshape(-1).repeat(self.num_agents).cpu().numpy(), rnn_hidden_state_q.cpu().numpy(), weights_prd.mean(dim=1).cpu().transpose(-1, -2).squeeze(0).numpy(), global_weights.reshape(-1).cpu().numpy(), Value.squeeze(0).cpu().numpy(), rnn_hidden_state_v.cpu().numpy()

			elif self.experiment_type in ["shared", "HAPPO"]:
				if "StarCraft" in self.environment:
					Value, _, _, rnn_hidden_state_v = self.target_critic_network_v(state_allies.to(self.device), state_enemies.to(self.device), one_hot_actions.to(self.device), rnn_hidden_state_v.to(self.device), indiv_masks.to(self.device))
				else:
					Value, _, _, rnn_hidden_state_v = self.target_critic_network_v(state_allies.to(self.device), None, one_hot_actions.to(self.device), rnn_hidden_state_v.to(self.device), indiv_masks.to(self.device))
			
				return None, None, None, Value.squeeze(0).cpu().numpy(), rnn_hidden_state_v.cpu().numpy()
			
			else:
				if "StarCraft" in self.environment:
					Value, _, _, rnn_hidden_state_v = self.target_critic_network_v(state_allies.to(self.device), state_enemies.to(self.device), one_hot_actions.to(self.device), rnn_hidden_state_v.to(self.device), indiv_masks.to(self.device))
					Q_value, _, weights_prd, _, _, rnn_hidden_state_q = self.target_critic_network_q(state_allies.to(self.device), state_enemies.to(self.device), one_hot_actions.to(self.device), rnn_hidden_state_q.to(self.device), indiv_masks.to(self.device))
				else:
					Value, _, _, rnn_hidden_state_v = self.target_critic_network_v(state_allies.to(self.device), None, one_hot_actions.to(self.device), rnn_hidden_state_v.to(self.device), indiv_masks.to(self.device))
					Q_value, _, weights_prd, _, _, rnn_hidden_state_q = self.target_critic_network_q(state_allies.to(self.device), None, one_hot_actions.to(self.device), rnn_hidden_state_q.to(self.device), indiv_masks.to(self.device))

				return Q_value.squeeze(0).cpu().numpy(), rnn_hidden_state_q.cpu().numpy(), torch.mean(weights_prd.transpose(-1, -2), dim=1).cpu().numpy(), Value.squeeze(0).cpu().numpy(), rnn_hidden_state_v.cpu().numpy()

	
	def update_epsilon(self):
		if self.warm_up:
			self.epsilon -= (self.epsilon_start - self.epsilon_end)/self.warm_up_episodes
			self.epsilon = max(self.epsilon, self.epsilon_end)

	
	def get_action(self, state_policy, mask_actions, hidden_state, greedy=False):
		with torch.no_grad():
			state_policy = torch.FloatTensor(state_policy).unsqueeze(0).unsqueeze(1).to(self.device)
			mask_actions = torch.BoolTensor(mask_actions).unsqueeze(0).unsqueeze(1).to(self.device)
			hidden_state = torch.FloatTensor(hidden_state).to(self.device)

			if self.experiment_type == "HAPPO":
				dists = []
				hidden_states = []
				for agent_id in range(self.num_agents):
					dists_i, hidden_state_i = self.policy_network[agent_id](state_policy[:, :, agent_id, :].unsqueeze(2), hidden_state[:, agent_id, :].unsqueeze(1), mask_actions[:, :, agent_id, :].unsqueeze(2))
					dists.append(dists_i)
					hidden_states.append(hidden_state_i)
				dists = torch.stack(dists, dim=0).squeeze(-2).permute(1, 2, 0, 3).float()
				hidden_states = torch.stack(hidden_states, dim=0).squeeze(-2).permute(1, 0, 2).float()
			else:
				dists, hidden_state = self.policy_network(state_policy, hidden_state, mask_actions)

			if self.warm_up:
				available_actions = (mask_actions>0).int()
				dists = (1.0-self.epsilon)*dists + available_actions*self.epsilon/torch.sum(available_actions, dim=-1).unsqueeze(-1)
			if greedy:
				actions = [dist.argmax().detach().cpu().item() for dist in dists.squeeze(0).squeeze(0)]
				action_logprob = None
			else:
				actions = [Categorical(dist).sample().detach().cpu().item() for dist in dists.squeeze(0).squeeze(0)]

				probs = Categorical(dists)
				action_logprob = probs.log_prob(torch.FloatTensor(actions).to(self.device)).cpu().numpy()

			return actions, action_logprob, hidden_state.cpu().numpy()



	def plot(self, masks, episode):
		
		self.comet_ml.log_metric('Policy_Loss',self.plotting_dict["policy_loss"],episode)
		self.comet_ml.log_metric('Grad_Norm_Policy',self.plotting_dict["grad_norm_policy"],episode)
		self.comet_ml.log_metric('Entropy',self.plotting_dict["entropy"],episode)

		# self.comet_ml.log_metric('Q_Value_LR',self.plotting_dict["q_value_lr"],episode)
		# self.comet_ml.log_metric('Policy_LR',self.plotting_dict["policy_lr"],episode)

		self.comet_ml.log_metric('V_Value_Loss',self.plotting_dict["v_value_loss"],episode)
		self.comet_ml.log_metric('Grad_Norm_V_Value',self.plotting_dict["grad_norm_value_v"],episode)
		# self.comet_ml.log_metric('V_Value_LR',self.plotting_dict["v_value_lr"],episode)
		
		# ENTROPY OF V WEIGHTS
		for i in range(self.num_heads):
			entropy_weights = -torch.sum(torch.sum((self.plotting_dict["weights_v"][:, i] * torch.log(torch.clamp(self.plotting_dict["weights_v"][:, i], 1e-10, 1.0)) * masks.view(-1, self.num_agents, 1)), dim=-1))/masks.sum()
			self.comet_ml.log_metric('V_Weight_Entropy_Head_'+str(i+1), entropy_weights.item(), episode)

		if "threshold" in self.experiment_type:
			for i in range(self.num_agents):
				agent_name = "agent"+str(i)
				self.comet_ml.log_metric('Group_Size_'+agent_name, self.plotting_dict["agent_groups_over_episode"][i].item(), episode)

			self.comet_ml.log_metric('Avg_Group_Size', self.plotting_dict["avg_agent_group_over_episode"].item(), episode)

		if "prd" in self.experiment_type:
			self.comet_ml.log_metric('Q_Value_Loss',self.plotting_dict["q_value_loss"],episode)
			self.comet_ml.log_metric('Grad_Norm_Q_Value',self.plotting_dict["grad_norm_value_q"],episode)

			# ENTROPY OF Q WEIGHTS
			for i in range(self.num_heads):
				entropy_weights = -torch.sum(torch.sum((self.plotting_dict["weights_prd"][:, i] * torch.log(torch.clamp(self.plotting_dict["weights_prd"][:, i], 1e-10, 1.0)) * masks.view(-1, self.num_agents, 1)), dim=-1))/masks.sum()
				self.comet_ml.log_metric('Q_Weight_Entropy_Head_'+str(i+1), entropy_weights.item(), episode)


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
		
		# v_value_lr, policy_lr = self.v_value_lr, self.policy_lr
		
		# if "prd" in self.experiment_type:
			# q_value_lr = self.q_value_lr
			# q_value_lr = self.lr_decay(episode, self.q_value_lr)
			# for param_group in self.q_critic_optimizer.param_groups:
			# 	param_group['lr'] = q_value_lr

		# v_value_lr = self.lr_decay(episode, self.v_value_lr)
		# for param_group in self.v_critic_optimizer.param_groups:
		# 	param_group['lr'] = v_value_lr

		# policy_lr = self.lr_decay(episode, self.policy_lr)
		# for param_group in self.policy_optimizer.param_groups:
		# 	param_group['lr'] = policy_lr

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

		self.buffer.calculate_targets(episode, self.select_above_threshold)

		
		# torch.autograd.set_detect_anomaly(True)
		# Optimize policy for n epochs
		for _ in range(self.n_epochs):

			# SAMPLE DATA FROM BUFFER
			states_critic_allies, states_critic_enemies, hidden_state_q, hidden_state_v, states_actor, hidden_state_actor, logprobs_old, \
			actions, last_one_hot_actions, one_hot_actions, action_masks, masks, values_old, target_values, q_values_old, target_q_values, advantage, factor  = self.buffer.sample_recurrent_policy()

			
			if self.norm_adv:
				shape = advantage.shape

				advantage_copy = copy.deepcopy(advantage)
				advantage_copy[masks.view(*shape) == 0.0] = float('nan')
				advantage_mean = torch.nanmean(advantage_copy)
				advantage_std = torch.from_numpy(np.array(np.nanstd(advantage_copy.cpu().numpy()))).float()

				advantage = ((advantage - advantage_mean) / (advantage_std + 1e-5))*masks.view(*shape)

			if "prd" in self.experiment_type:

				q_values_old *= masks
				target_shape = q_values_old.shape
				
				if "StarCraft" in self.environment:
					q_values, _, weights_prd, _, score_q, _ = self.critic_network_q(
														states_critic_allies.to(self.device),
														states_critic_enemies.to(self.device),
														one_hot_actions.to(self.device),
														hidden_state_q.to(self.device),
														masks.to(self.device),
														)
				else:
					q_values, _, weights_prd, _, score_q, _ = self.critic_network_q(
														states_critic_allies.to(self.device),
														None,
														one_hot_actions.to(self.device),
														hidden_state_q.to(self.device),
														masks.to(self.device),
														)

				# because q_value returned by the model is one single value for each timesteps but it expects N (num_agents) q_values for each timestep
				if self.experiment_type == "prd_soft_advantage_global":
					q_values = q_values.reshape(target_shape[0], target_shape[1], 1).repeat(1, 1, self.num_agents)

				q_values = q_values.reshape(*target_shape)

				q_values *= masks.to(self.device)
				target_q_values *= masks

				critic_q_loss_1 = F.huber_loss(q_values, target_q_values.to(self.device), reduction="sum", delta=10.0) / masks.sum()
				critic_q_loss_2 = F.huber_loss(torch.clamp(q_values, q_values_old.to(self.device)-self.value_clip, q_values_old.to(self.device)+self.value_clip), target_q_values.to(self.device), reduction="sum", delta=10.0) / masks.sum()
				# critic_q_loss = F.huber_loss(q_values*masks.to(self.device), target_q_values.to(self.device)*masks.to(self.device), reduction="sum", delta=10.0) / masks.sum() #(self.num_agents*masks.sum())

			

			values_old *= masks

			target_shape = values_old.shape
			if "StarCraft" in self.environment:
				values, weight_v, score_v, h_v = self.critic_network_v(
													states_critic_allies.to(self.device),
													states_critic_enemies.to(self.device),
													one_hot_actions.to(self.device),
													hidden_state_v.to(self.device),
													masks.to(self.device),
													)
			else:
				values, weight_v, score_v, h_v = self.critic_network_v(
													states_critic_allies.to(self.device),
													None,
													one_hot_actions.to(self.device),
													hidden_state_v.to(self.device),
													masks.to(self.device),
													)
			values = values.reshape(*target_shape)

			values *= masks.to(self.device)
			target_values *= masks

			critic_v_loss_1 = F.huber_loss(values, target_values.to(self.device), reduction="sum", delta=10.0) / masks.sum()
			critic_v_loss_2 = F.huber_loss(torch.clamp(values, values_old.to(self.device)-self.value_clip, values_old.to(self.device)+self.value_clip), target_values.to(self.device), reduction="sum", delta=10.0) / masks.sum()
			# critic_v_loss = F.huber_loss(values*masks.to(self.device), target_values.to(self.device)*masks.to(self.device), reduction="sum", delta=10.0) / masks.sum() #(self.num_agents*masks.sum())

			
			dists, _ = self.policy_network(
					states_actor.to(self.device),
					hidden_state_actor.to(self.device),
					action_masks.to(self.device),
					)

			probs = Categorical(dists)
			logprobs = probs.log_prob(actions.to(self.device))
			
			if "threshold" in self.experiment_type or "top" in self.experiment_type:
				mask_rewards = (weights_prd.transpose(-1, -2)>self.select_above_threshold).int()
				agent_groups_over_episode = torch.sum(torch.sum(mask_rewards.reshape(-1, self.num_agents, self.num_agents).float(), dim=-2),dim=0)/masks.sum()
				avg_agent_group_over_episode = torch.mean(agent_groups_over_episode)
				agent_groups_over_episode_batch += agent_groups_over_episode
				avg_agent_group_over_episode_batch += avg_agent_group_over_episode

			entropy_weights_v = 0
			# score_v_cum = 0
			if "prd" in self.experiment_type:
				entropy_weights = 0
				# score_q_cum = 0
			
			for i in range(self.num_heads):
				entropy_weights_v += -torch.sum(torch.sum(weight_v[:, i] * torch.log(torch.clamp(weight_v[:, i], 1e-10, 1.0)) * masks.view(-1, self.num_agents, 1).to(self.device), dim=-1))/masks.sum()
				# score_v_cum += (score_v[:, i].squeeze(-2)**2 * masks.view(-1, self.num_agents, 1).to(self.device)).sum()/masks.sum()

				if "prd" in self.experiment_type:
					entropy_weights += -torch.sum(torch.sum((weights_prd[:, i] * torch.log(torch.clamp(weights_prd[:, i], 1e-10, 1.0)) * masks.view(-1, self.num_agents, 1).to(self.device)), dim=-1))/masks.sum()		
					# score_q_cum += (score_q[:, i].squeeze(-2)**2 * masks.view(-1, self.num_agents, 1).to(self.device)).sum()/masks.sum()
			
			critic_v_loss = torch.max(critic_v_loss_1, critic_v_loss_2) #+ self.critic_score_regularizer*score_v_cum + self.critic_weight_entropy_pen*entropy_weights_v
			print("Critic V Loss", critic_v_loss.item())
			
			if "prd" in self.experiment_type:
				critic_q_loss = torch.max(critic_q_loss_1, critic_q_loss_2) #+ self.critic_score_regularizer*score_q_cum + self.critic_weight_entropy_pen*entropy_weights
				print("Critic Q Loss", critic_q_loss.item())
				

			# Finding the ratio (pi_theta / pi_theta__old)
			ratios = torch.exp((logprobs - logprobs_old.to(self.device)))
			
			# Finding Surrogate Loss
			surr1 = ratios * advantage.to(self.device) * masks.to(self.device)
			surr2 = torch.clamp(ratios, 1-self.policy_clip, 1+self.policy_clip) * advantage.to(self.device) * masks.to(self.device)

			# final loss of clipped objective PPO
			entropy = -torch.sum(torch.sum(dists*masks.unsqueeze(-1).to(self.device) * torch.log(torch.clamp(dists*masks.unsqueeze(-1).to(self.device), 1e-10,1.0)), dim=-1))/ masks.sum() #(masks.sum()*self.num_agents)
			policy_loss_ = (-torch.min(surr1, surr2).sum())/masks.sum()
			policy_loss = policy_loss_ - self.entropy_pen*entropy

			print("Policy Loss", policy_loss_.item(), "Entropy", (-self.entropy_pen*entropy.item()))

			if "prd" in self.experiment_type:
				self.q_critic_optimizer.zero_grad()
				critic_q_loss.backward()
				if self.enable_grad_clip_critic_q:
					grad_norm_value_q = torch.nn.utils.clip_grad_norm_(self.critic_network_q.parameters(), self.grad_clip_critic_q)
				else:
					total_norm = 0
					for p in self.critic_network_q.parameters():
						param_norm = p.grad.detach().data.norm(2)
						total_norm += param_norm.item() ** 2
					grad_norm_value_q = torch.tensor([total_norm ** 0.5])
				self.q_critic_optimizer.step()
			
			
			self.v_critic_optimizer.zero_grad()
			critic_v_loss.backward()
			if self.enable_grad_clip_critic_v:
				grad_norm_value_v = torch.nn.utils.clip_grad_norm_(self.critic_network_v.parameters(), self.grad_clip_critic_v)
			else:
				total_norm = 0
				for p in self.critic_network_v.parameters():
					param_norm = p.grad.detach().data.norm(2)
					total_norm += param_norm.item() ** 2
				grad_norm_value_v = torch.tensor([total_norm ** 0.5])
			self.v_critic_optimizer.step()

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

			
			policy_loss_batch += policy_loss.item()
			entropy_batch += entropy.item()
			grad_norm_policy_batch += grad_norm_policy
			v_value_loss_batch += critic_v_loss.item()
			grad_norm_value_v_batch += grad_norm_value_v
			if weight_v_batch is None:
				weight_v_batch = weight_v.detach().cpu()
			else:
				weight_v_batch += weight_v.detach().cpu()

			if "prd" in self.experiment_type:
				q_value_loss_batch += critic_q_loss.item()
				grad_norm_value_q_batch += grad_norm_value_q
				if weight_prd_batch is None:
					weight_prd_batch = weights_prd.detach().cpu()
				else:
					weight_prd_batch += weights_prd.detach().cpu()
			
			
			

		# Copy new weights into old critic
		if "prd" in self.experiment_type:
			if self.soft_update_q:
				for target_param, param in zip(self.target_critic_network_q.parameters(), self.critic_network_q.parameters()):
					target_param.data.copy_(target_param.data * (1.0 - self.tau_q) + param.data * self.tau_q)
			elif episode % self.network_update_interval_q == 0:
				self.target_critic_network_q.load_state_dict(self.critic_network_q.state_dict())
			
		
		if self.soft_update_v:
			for target_param, param in zip(self.target_critic_network_v.parameters(), self.critic_network_v.parameters()):
				target_param.data.copy_(target_param.data * (1.0 - self.tau_v) + param.data * self.tau_v)
		elif episode % self.network_update_interval_v == 0:
			self.target_critic_network_v.load_state_dict(self.critic_network_v.state_dict())

		# self.scheduler.step()
		# print("learning rate of policy", self.scheduler.get_lr())

		# clear buffer
		self.buffer.clear()
		

		policy_loss_batch /= self.n_epochs
		entropy_batch /= self.n_epochs
		grad_norm_policy_batch /= self.n_epochs
		v_value_loss_batch /= self.n_epochs
		grad_norm_value_v_batch /= self.n_epochs
		weight_v_batch /= self.n_epochs
		agent_groups_over_episode_batch /= self.n_epochs
		avg_agent_group_over_episode_batch /= self.n_epochs

		if "prd" in self.experiment_type:
			q_value_loss_batch /= self.n_epochs
			grad_norm_value_q_batch /= self.n_epochs
			weight_prd_batch /= self.n_epochs
			


		if "prd" in self.experiment_type:
			if self.update_learning_rate_with_prd:
				for g in self.policy_optimizer.param_groups:
					g['lr'] = self.policy_lr * self.num_agents/avg_agent_group_over_episode_batch


		self.plotting_dict = {
		"v_value_loss": v_value_loss_batch,
		"policy_loss": policy_loss_batch,
		"entropy": entropy_batch,
		"grad_norm_value_v": grad_norm_value_v_batch,
		"grad_norm_policy": grad_norm_policy_batch,
		"weights_v": weight_v_batch,
		# "q_value_lr": q_value_lr,
		# "policy_lr": policy_lr,
		}

		if "prd" in self.experiment_type:
			self.plotting_dict["q_value_loss"] = q_value_loss_batch
			self.plotting_dict["grad_norm_value_q"] = grad_norm_value_q_batch
			self.plotting_dict["weights_prd"] = weight_prd_batch
			# self.plotting_dict["v_value_lr"] = v_value_lr

		if "threshold" in self.experiment_type:
			self.plotting_dict["agent_groups_over_episode"] = agent_groups_over_episode_batch
			self.plotting_dict["avg_agent_group_over_episode"] = avg_agent_group_over_episode_batch
		if "prd_top" in self.experiment_type:
			self.plotting_dict["mean_min_weight_value"] = mean_min_weight_value

		if self.comet_ml is not None:
			self.plot(masks, episode)

		del q_value_loss_batch, v_value_loss_batch, policy_loss_batch, entropy_batch, grad_norm_value_v_batch, grad_norm_value_q_batch, grad_norm_policy_batch, weight_prd_batch, agent_groups_over_episode_batch, avg_agent_group_over_episode_batch
		torch.cuda.empty_cache()



	def update_HAPPO(self, episode):


		self.buffer.calculate_targets(episode, self.select_above_threshold)
		
		# torch.autograd.set_detect_anomaly(True)
		# Optimize policy for n epochs
		agent_permutation = torch.randperm(self.num_agents)

		data_chunks = self.max_time_steps//self.data_chunk_length
		self.buffer.factor = torch.ones((self.update_ppo_agent*data_chunks, self.data_chunk_length)).float()

		avg_policy_loss = 0
		avg_policy_grad_norm = 0
		avg_policy_entropy = 0
		avg_critic_loss = 0
		avg_critic_grad_norm = 0
		avg_attention_weights_v = None

		train_critic = True # train critic once because its parameters are shared

		for agent_id in agent_permutation:

			for _ in range(self.n_epochs):

				# SAMPLE DATA FROM BUFFER
				states_critic_allies, states_critic_enemies, hidden_state_q, hidden_state_v, states_actor, hidden_state_actor, logprobs_old, \
				actions, last_one_hot_actions, one_hot_actions, action_masks, masks, values_old, target_values, q_values_old, target_q_values, advantage, factor  = self.buffer.sample_recurrent_policy()

				if train_critic:
					values_old *= masks

					target_shape = q_values_old.shape
					if "StarCraft" in self.environment:
						values, attention_weights_v, score_v, _ = self.critic_network_v(
															states_critic_allies.to(self.device),
															states_critic_enemies.to(self.device),
															one_hot_actions.to(self.device),
															hidden_state_q.to(self.device),
															masks.to(self.device),
															)
					else:
						values, attention_weights_v, score_v, _ = self.critic_network_v(
															states_critic_allies.to(self.device),
															None,
															one_hot_actions.to(self.device),
															hidden_state_q.to(self.device),
															masks.to(self.device),
															)

					values = values.reshape(*target_shape)

					values *= masks.to(self.device)	
					target_values *= masks

					critic_v_loss_1 = F.huber_loss(values, target_values.to(self.device), reduction="sum", delta=10.0) / masks.sum()
					critic_v_loss_2 = F.huber_loss(torch.clamp(values, values_old.to(self.device)-self.value_clip, values_old.to(self.device)+self.value_clip), target_values.to(self.device), reduction="sum", delta=10.0) / masks.sum()

					entropy_weights = 0
					score_v_cum = 0
					for i in range(self.num_heads):
						entropy_weights += -torch.sum(torch.sum((attention_weights_v[:, i] * torch.log(torch.clamp(attention_weights_v[:, i], 1e-10, 1.0)) * masks.view(-1, self.num_agents, 1).to(self.device)), dim=-1))/masks.sum()
						score_v_cum += (score_v[:, i].squeeze(-2)**2 * masks.view(-1, self.num_agents, 1).to(self.device)).sum()/masks.sum()
					
					critic_v_loss = torch.max(critic_v_loss_1, critic_v_loss_2) + self.critic_score_regularizer*score_v_cum + self.critic_weight_entropy_pen*entropy_weights

					self.v_critic_optimizer.zero_grad()
					critic_v_loss.backward()
					if self.enable_grad_clip_critic_v:
						grad_norm_value_v = torch.nn.utils.clip_grad_norm_(self.critic_network_v.parameters(), self.grad_clip_critic_v)
					else:
						total_norm = 0
						for p in self.critic_network_v.parameters():
							param_norm = p.grad.detach().data.norm(2)
							total_norm += param_norm.item() ** 2
						grad_norm_value_v = torch.tensor([total_norm ** 0.5])
					self.v_critic_optimizer.step()

				if self.norm_adv:
					shape = advantage.shape
					advantage_copy = copy.deepcopy(advantage)
					advantage_copy[masks.view(*shape) == 0.0] = float('nan')
					advantage_mean = torch.nanmean(advantage_copy)
					advantage_std = torch.from_numpy(np.array(np.nanstd(advantage_copy.cpu().numpy()))).float()
					
					advantage = ((advantage - advantage_mean) / (advantage_std + 1e-5))*masks.view(*shape)


			
				shape = hidden_state_actor.shape
				hidden_state_actor = hidden_state_actor.reshape(shape[0], -1, self.num_agents, self.rnn_hidden_actor)
			
			
				dists, _ = self.policy_network[agent_id](
							states_actor[:, :, agent_id, :].to(self.device).unsqueeze(-2),
							hidden_state_actor[:, :, agent_id, :].to(self.device).unsqueeze(-2),
							action_masks[:, :, agent_id, :].to(self.device).unsqueeze(-2),
							)

				probs = Categorical(dists.squeeze(-2))

				logprobs = probs.log_prob(actions[:, :, agent_id].to(self.device))

				ratios = torch.exp((logprobs - logprobs_old[:, :, agent_id].to(self.device)))

				# Finding Surrogate Loss
				surr1 = ratios * advantage[:, :, agent_id].to(self.device) * masks[:, :, agent_id].to(self.device)
				surr2 = torch.clamp(ratios, 1-self.policy_clip, 1+self.policy_clip) * advantage[:, :, agent_id].to(self.device) * masks[:, :, agent_id].to(self.device)

				# final loss of clipped objective PPO
				entropy = -torch.sum(torch.sum(dists.squeeze(-2)*masks[:, :, agent_id].unsqueeze(-1).to(self.device) * torch.log(torch.clamp(dists.squeeze(-2)*masks[:, :, agent_id].unsqueeze(-1).to(self.device), 1e-10,1.0)), dim=-1))/ masks[:, :, agent_id].sum()
				policy_loss_ = ((-torch.min(surr1, surr2)*factor.to(self.device)*masks[:, :, agent_id].to(self.device)).sum())/masks[:, :, agent_id].sum()
				policy_loss = policy_loss_ - self.entropy_pen*entropy

				self.policy_optimizer[agent_id].zero_grad()
				policy_loss.backward()
				if self.enable_grad_clip_actor:
					grad_norm_policy = torch.nn.utils.clip_grad_norm_(self.policy_network[agent_id].parameters(), self.grad_clip_actor)
				else:
					total_norm = 0
					for p in self.policy_network[agent_id].parameters():
						param_norm = p.grad.detach().data.norm(2)
						total_norm += param_norm.item() ** 2
					grad_norm_policy = torch.tensor([total_norm ** 0.5])
				self.policy_optimizer[agent_id].step()

				

				avg_policy_loss += policy_loss.item()
				avg_policy_entropy += entropy.item()
				avg_policy_grad_norm += grad_norm_policy.item()

				if train_critic:
					avg_critic_loss += critic_v_loss.item()
					avg_critic_grad_norm += grad_norm_value_v.item()
					if avg_attention_weights_v is None:
						avg_attention_weights_v = attention_weights_v.detach().cpu()
					else:
						avg_attention_weights_v += attention_weights_v.detach().cpu()

				# del dists, probs, logprobs, ratios, surr1, surr2, entropy, policy_loss_, policy_loss

			with torch.no_grad():
				# POST UPDATE POLICY OUTPUT TO UPDATE FACTOR
				states_actor_ = torch.from_numpy(self.buffer.states_actor).reshape(self.update_ppo_agent, data_chunks, self.data_chunk_length, self.num_agents, -1).float()[:, :, :, agent_id, :].unsqueeze(-2).reshape(self.update_ppo_agent*data_chunks, self.data_chunk_length, 1, -1)
				hidden_state_actor_ = torch.from_numpy(self.buffer.hidden_state_actor).float().reshape(self.update_ppo_agent, data_chunks, self.data_chunk_length, self.rnn_num_layers_actor, self.num_agents, -1)[:, :, 0, :, agent_id, :].unsqueeze(-2).permute(2, 0, 1, 3, 4).reshape(self.rnn_num_layers_actor, -1, self.rnn_hidden_actor)
				action_masks = torch.from_numpy(self.buffer.action_masks).bool().reshape(self.update_ppo_agent, data_chunks, self.data_chunk_length, self.num_agents, -1)[:, :, :, agent_id, :].unsqueeze(-2).reshape(self.update_ppo_agent*data_chunks, self.data_chunk_length, 1, -1)
				dists_new, _ = self.policy_network[agent_id](
							states_actor_.to(self.device),
							hidden_state_actor_.to(self.device),
							action_masks.to(self.device),
							)

				probs_new = Categorical(dists_new.squeeze(-2))

				logprobs_new = probs_new.log_prob(actions[:, :, agent_id].to(self.device))

			target_shape = self.buffer.factor.shape
			self.buffer.factor = self.buffer.factor*torch.exp((logprobs_new.reshape(*target_shape).cpu()-torch.from_numpy(self.buffer.logprobs[:, :, agent_id]).float().reshape(*target_shape))*masks[:, :, agent_id].reshape(*target_shape).to(self.device)).reshape(self.update_ppo_agent*data_chunks, self.data_chunk_length).detach()

			train_critic = False

		# Copy new weights into old critic
		if self.soft_update_v:
			for target_param, param in zip(self.target_critic_network_v.parameters(), self.critic_network_v.parameters()):
				target_param.data.copy_(target_param.data * (1.0 - self.tau_v) + param.data * self.tau_v)
		elif episode % self.network_update_interval_v == 0:
			self.target_critic_network_v.load_state_dict(self.critic_network_v.state_dict())

		# clear buffer
		self.buffer.clear()
		

		avg_policy_loss /= (self.n_epochs*self.num_agents)
		avg_policy_entropy /= (self.n_epochs*self.num_agents)
		avg_policy_grad_norm /= (self.n_epochs*self.num_agents)
		avg_critic_loss /= self.n_epochs
		avg_critic_grad_norm /= self.n_epochs
		avg_attention_weights_v /= self.n_epochs


		
		self.plotting_dict = {
		"v_value_loss": avg_critic_loss,
		"policy_loss": avg_policy_loss,
		"entropy": avg_policy_entropy,
		"grad_norm_value_v": avg_critic_grad_norm,
		"grad_norm_policy": avg_policy_grad_norm,
		"weights_v": avg_attention_weights_v,
		}

		if self.comet_ml is not None:
			self.plot(masks, episode)

		del avg_critic_loss, avg_policy_loss, avg_policy_entropy, avg_critic_grad_norm, avg_policy_grad_norm, avg_attention_weights_v
		torch.cuda.empty_cache()
