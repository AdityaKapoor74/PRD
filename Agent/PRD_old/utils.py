import torch
from torch import Tensor
import numpy as np
import random

# from ppo_model import RunningMeanStd


class RolloutBuffer:
	def __init__(
		self, 
		num_episodes, 
		max_time_steps, 
		num_agents, 
		num_enemies,
		obs_shape_critic_ally, 
		obs_shape_critic_enemy, 
		obs_shape_actor, 
		rnn_num_layers_actor,
		actor_hidden_state_dim,
		rnn_num_layers_v,
		v_hidden_state_dim,
		num_actions, 
		data_chunk_length,
		norm_returns_v,
		clamp_rewards,
		clamp_rewards_value_min,
		clamp_rewards_value_max,
		norm_rewards,
		target_calc_style,
		td_lambda,
		gae_lambda,
		n_steps,
		gamma,
		# V_PopArt,
		):
		self.num_episodes = num_episodes
		self.max_time_steps = max_time_steps
		self.num_agents = num_agents
		self.num_enemies = num_enemies
		self.obs_shape_critic_ally = obs_shape_critic_ally
		self.obs_shape_critic_enemy = obs_shape_critic_enemy
		self.obs_shape_actor = obs_shape_actor
		self.rnn_num_layers_actor = rnn_num_layers_actor
		self.actor_hidden_state_dim = actor_hidden_state_dim
		self.rnn_num_layers_v = rnn_num_layers_v
		self.v_hidden_state_dim = v_hidden_state_dim
		self.num_actions = num_actions

		self.data_chunk_length = data_chunk_length
		self.norm_returns_v = norm_returns_v
		self.clamp_rewards = clamp_rewards
		self.clamp_rewards_value_min = clamp_rewards_value_min
		self.clamp_rewards_value_max = clamp_rewards_value_max
		self.norm_rewards = norm_rewards

		self.target_calc_style = target_calc_style
		self.td_lambda = td_lambda
		self.gae_lambda = gae_lambda
		self.gamma = gamma
		self.n_steps = n_steps
			
		# if self.norm_returns_v:
		# 	self.v_value_norm = V_PopArt

		# if self.norm_rewards:
		# 	self.reward_norm = RunningMeanStd(shape=(1), device=self.device)

		self.episode_num = 0
		self.time_step = 0

		self.states_critic_allies = np.zeros((num_episodes, max_time_steps, num_agents, obs_shape_critic_ally))
		self.states_critic_enemies = np.zeros((num_episodes, max_time_steps, num_enemies, obs_shape_critic_enemy))
		self.hidden_state_v = np.zeros((num_episodes, max_time_steps, rnn_num_layers_v, num_agents*num_agents, v_hidden_state_dim))
		self.Values = np.zeros((num_episodes, max_time_steps+1, num_agents, num_agents))
		self.weights_prd = np.zeros((num_episodes, max_time_steps, num_agents, num_agents))
		self.states_actor = np.zeros((num_episodes, max_time_steps, num_agents, obs_shape_actor))
		self.hidden_state_actor = np.zeros((num_episodes, max_time_steps, rnn_num_layers_actor, num_agents, actor_hidden_state_dim))
		self.logprobs = np.zeros((num_episodes, max_time_steps, num_agents))
		self.actions = np.zeros((num_episodes, max_time_steps, num_agents), dtype=int)
		self.last_one_hot_actions = np.zeros((num_episodes, max_time_steps, num_agents, num_actions))
		self.action_probs = np.zeros((num_episodes, max_time_steps, num_agents, num_actions))
		self.one_hot_actions = np.zeros((num_episodes, max_time_steps, num_agents, num_actions))
		self.action_masks = np.zeros((num_episodes, max_time_steps, num_agents, num_actions))
		self.rewards = np.zeros((num_episodes, max_time_steps, num_agents))
		self.dones = np.ones((num_episodes, max_time_steps+1, num_agents))

		self.episode_length = np.zeros(num_episodes)
	

	def clear(self):

		self.states_critic_allies = np.zeros((self.num_episodes, self.max_time_steps, self.num_agents, self.obs_shape_critic_ally))
		self.states_critic_enemies = np.zeros((self.num_episodes, self.max_time_steps, self.num_enemies, self.obs_shape_critic_enemy))
		self.hidden_state_v = np.zeros((self.num_episodes, self.max_time_steps, self.rnn_num_layers_v, self.num_agents*self.num_agents, self.v_hidden_state_dim))
		self.Values = np.zeros((self.num_episodes, self.max_time_steps+1, self.num_agents, self.num_agents))
		self.weights_prd = np.zeros((self.num_episodes, self.max_time_steps, self.num_agents, self.num_agents))
		self.states_actor = np.zeros((self.num_episodes, self.max_time_steps, self.num_agents, self.obs_shape_actor))
		self.hidden_state_actor = np.zeros((self.num_episodes, self.max_time_steps, self.rnn_num_layers_actor, self.num_agents, self.actor_hidden_state_dim))
		self.logprobs = np.zeros((self.num_episodes, self.max_time_steps, self.num_agents))
		self.actions = np.zeros((self.num_episodes, self.max_time_steps, self.num_agents), dtype=int)
		self.last_one_hot_actions = np.zeros((self.num_episodes, self.max_time_steps, self.num_agents, self.num_actions))
		self.action_probs = np.zeros((self.num_episodes, self.max_time_steps, self.num_agents, self.num_actions))
		self.one_hot_actions = np.zeros((self.num_episodes, self.max_time_steps, self.num_agents, self.num_actions))
		self.action_masks = np.zeros((self.num_episodes, self.max_time_steps, self.num_agents, self.num_actions))
		self.rewards = np.zeros((self.num_episodes, self.max_time_steps, self.num_agents))
		self.dones = np.ones((self.num_episodes, self.max_time_steps+1, self.num_agents))

		self.episode_length = np.zeros(self.num_episodes)

		self.time_step = 0
		self.episode_num = 0

	def push(
		self, 
		state_critic_allies, 
		state_critic_enemies, 
		values,
		hidden_state_v,
		weights_prd,
		state_actor, 
		hidden_state_actor, 
		logprobs, 
		actions, 
		action_probs,
		last_one_hot_actions, 
		one_hot_actions, 
		action_masks, 
		rewards, 
		dones
		):

		self.states_critic_allies[self.episode_num][self.time_step] = state_critic_allies
		self.states_critic_enemies[self.episode_num][self.time_step] = state_critic_enemies
		self.hidden_state_v[self.episode_num][self.time_step] = hidden_state_v
		self.Values[self.episode_num][self.time_step] = values
		self.weights_prd[self.episode_num][self.time_step] = weights_prd
		self.states_actor[self.episode_num][self.time_step] = state_actor
		self.hidden_state_actor[self.episode_num][self.time_step] = hidden_state_actor
		self.logprobs[self.episode_num][self.time_step] = logprobs
		self.actions[self.episode_num][self.time_step] = actions
		self.action_probs[self.episode_num][self.time_step] = action_probs
		self.last_one_hot_actions[self.episode_num][self.time_step] = last_one_hot_actions
		self.one_hot_actions[self.episode_num][self.time_step] = one_hot_actions
		self.action_masks[self.episode_num][self.time_step] = action_masks
		self.rewards[self.episode_num][self.time_step] = rewards
		self.dones[self.episode_num][self.time_step] = dones

		if self.time_step < self.max_time_steps-1:
			self.time_step += 1


	def end_episode(
		self, 
		t, 
		value, 
		dones
		):
		self.Values[self.episode_num][self.time_step+1] = value
		self.dones[self.episode_num][self.time_step+1] = dones

		self.episode_length[self.episode_num] = t
		self.episode_num += 1
		self.time_step = 0


	def sample_recurrent_policy(self):

		data_chunks = self.max_time_steps // self.data_chunk_length
		rand_batch = np.random.permutation(self.num_episodes)
		rand_time = np.random.permutation(data_chunks)

		states_critic_allies = torch.from_numpy(self.states_critic_allies).float().reshape(self.num_episodes, data_chunks, self.data_chunk_length, self.num_agents, self.obs_shape_critic_ally)[:, rand_time][rand_batch, :].reshape(-1, self.data_chunk_length, self.num_agents, self.obs_shape_critic_ally)
		states_critic_enemies = torch.from_numpy(self.states_critic_enemies).float().reshape(self.num_episodes, data_chunks, self.data_chunk_length, self.num_enemies, self.obs_shape_critic_enemy)[:, rand_time][rand_batch, :].reshape(-1, self.data_chunk_length, self.num_enemies, self.obs_shape_critic_enemy)
		hidden_state_v = torch.from_numpy(self.hidden_state_v).float().reshape(self.num_episodes, data_chunks, self.data_chunk_length, self.rnn_num_layers_v, self.num_agents*self.num_agents, self.v_hidden_state_dim)[:, rand_time][rand_batch, :][:, :, 0, :, :, :].permute(2, 0, 1, 3, 4).reshape(self.rnn_num_layers_v, -1, self.v_hidden_state_dim)
		states_actor = torch.from_numpy(self.states_actor).float().reshape(self.num_episodes, data_chunks, self.data_chunk_length, self.num_agents, self.obs_shape_actor)[:, rand_time][rand_batch, :].reshape(-1, self.data_chunk_length, self.num_agents, self.obs_shape_actor).reshape(-1, self.data_chunk_length, self.num_agents, self.obs_shape_actor)
		hidden_state_actor = torch.from_numpy(self.hidden_state_actor).float().reshape(self.num_episodes, data_chunks, self.data_chunk_length, self.rnn_num_layers_actor, self.num_agents, self.actor_hidden_state_dim)[:, rand_time][rand_batch, :][:, :, 0, :, :, :].permute(2, 0, 1, 3, 4).reshape(self.rnn_num_layers_actor, -1, self.actor_hidden_state_dim)
		logprobs = torch.from_numpy(self.logprobs).float().reshape(self.num_episodes, data_chunks, self.data_chunk_length, self.num_agents)[:, rand_time][rand_batch, :].reshape(-1, self.data_chunk_length, self.num_agents)
		actions = torch.from_numpy(self.actions).long().reshape(self.num_episodes, data_chunks, self.data_chunk_length, self.num_agents)[:, rand_time][rand_batch, :].reshape(-1, self.data_chunk_length, self.num_agents)
		action_probs = torch.from_numpy(self.action_probs).float().reshape(self.num_episodes, data_chunks, self.data_chunk_length, self.num_agents, self.num_actions)[:, rand_time][rand_batch, :].reshape(-1, self.data_chunk_length, self.num_agents, self.num_actions)
		last_one_hot_actions = torch.from_numpy(self.last_one_hot_actions).float().reshape(self.num_episodes, data_chunks, self.data_chunk_length, self.num_agents, self.num_actions)[:, rand_time][rand_batch, :].reshape(-1, self.data_chunk_length, self.num_agents, self.num_actions)
		one_hot_actions = torch.from_numpy(self.one_hot_actions).float().reshape(self.num_episodes, data_chunks, self.data_chunk_length, self.num_agents, self.num_actions)[:, rand_time][rand_batch, :].reshape(-1, self.data_chunk_length, self.num_agents, self.num_actions)
		action_masks = torch.from_numpy(self.action_masks).bool().reshape(self.num_episodes, data_chunks, self.data_chunk_length, self.num_agents, self.num_actions)[:, rand_time][rand_batch, :].reshape(-1, self.data_chunk_length, self.num_agents, self.num_actions)
		masks = 1-torch.from_numpy(self.dones[:, :-1]).float().reshape(self.num_episodes, data_chunks, self.data_chunk_length, self.num_agents)[:, rand_time][rand_batch, :].reshape(-1, self.data_chunk_length, self.num_agents)
		
		# target_values, target_q_values, advantage = self.calculate_targets(advantage_type, episode, select_above_threshold)
		values = torch.from_numpy(self.Values[:, :-1, :]).float().reshape(self.num_episodes, data_chunks, self.data_chunk_length, self.num_agents, self.num_agents)[:, rand_time][rand_batch, :].reshape(-1, self.data_chunk_length, self.num_agents, self.num_agents)
		target_values = self.target_values.float().reshape(self.num_episodes, data_chunks, self.data_chunk_length, self.num_agents, self.num_agents)[:, rand_time][rand_batch, :].reshape(-1, self.data_chunk_length, self.num_agents, self.num_agents)
		advantage = self.advantage.float().reshape(self.num_episodes, data_chunks, self.data_chunk_length, self.num_agents)[:, rand_time][rand_batch, :].reshape(-1, self.data_chunk_length, self.num_agents)
		
		return states_critic_allies, states_critic_enemies, hidden_state_v, states_actor, hidden_state_actor, logprobs, \
		actions, action_probs, last_one_hot_actions, one_hot_actions, action_masks, masks, values, target_values, advantage

	def calculate_targets(self, advantage_type, episode, select_above_threshold, v_value_norm):
		
		masks = 1 - torch.from_numpy(self.dones[:, :-1, :])
		next_mask = 1 - torch.from_numpy(self.dones[:, -1, :])

		rewards = torch.from_numpy(self.rewards)

		if self.clamp_rewards:
			rewards = torch.clamp(rewards, min=self.clamp_rewards_value_min, max=self.clamp_rewards_value_max)

		if self.norm_rewards:
			self.reward_norm.update(rewards.view(-1).to(self.device), masks.view(-1).to(self.device))
			rewards = ((rewards.to(self.device) - self.reward_norm.mean) / (torch.sqrt(self.reward_norm.var) + 1e-5)).cpu().view(-1, self.num_agents)
		
		# TARGET CALC
		values = torch.from_numpy(self.Values[:, :-1, :, :]) * masks.unsqueeze(-2).repeat(1, 1, self.num_agents, 1)
		next_values = torch.from_numpy(self.Values[:, -1, :, :]) * next_mask.unsqueeze(-2).repeat(1, 1, self.num_agents, 1)

		if self.norm_returns_v:
			values_shape = values.shape
			values = v_value_norm.denormalize(values.view(-1)).view(values_shape) * masks.unsqueeze(-2).repeat(1, 1, self.num_agents, 1)

			next_values_shape = next_values.shape
			next_values = v_value_norm.denormalize(next_values.view(-1)).view(next_values_shape) * next_mask.unsqueeze(-2).repeat(1, 1, self.num_agents, 1)

		if self.target_calc_style == "GAE":
			target_values = self.gae_targets(rewards.unsqueeze(-2).repeat(1, 1, self.num_agents, 1), values, next_values, masks.unsqueeze(-2).repeat(1, 1, self.num_agents, 1), next_mask.unsqueeze(-2).repeat(1, 1, self.num_agents, 1))
		elif self.target_calc_style == "N_steps":
			target_values = self.nstep_returns(rewards.unsqueeze(-2).repeat(1, 1, self.num_agents, 1), values, next_values, masks.unsqueeze(-2).repeat(1, 1, self.num_agents, 1), next_mask.unsqueeze(-2).repeat(1, 1, self.num_agents, 1))

		self.advantage = (target_values - values).transpose(-1, -2)

		if "prd_above_threshold_ascend" in advantage_type or "prd_above_threshold_decay" in advantage_type:
			masking_advantages = (torch.from_numpy(self.weights_prd).float()>select_above_threshold).int()
			self.advantage = (self.advantage * masking_advantages).sum(dim=-1)
		else:
			self.advantage = self.advantage.sum(dim=-1)

		# if self.norm_returns_v:
		# 	targets_shape = target_values.shape
		# 	v_value_norm.update(target_values.view(-1), masks.unsqueeze(-2).repeat(1, 1, self.num_agents, 1).view(-1))
			
		# 	target_values = v_value_norm.normalize(target_values.view(-1)).view(targets_shape) * masks.unsqueeze(-2).repeat(1, 1, self.num_agents, 1)
		
		self.target_values = target_values.cpu()


	def gae_targets(self, rewards, values, next_value, masks, next_mask):
		
		target_values = rewards.new_zeros(*rewards.shape)
		advantage = 0

		for t in reversed(range(0, rewards.shape[1])):

			td_error = rewards[:,t,:,:] + (self.gamma * next_value * next_mask) - values.data[:,t,:,:] * masks[:, t, :, :]
			advantage = td_error + self.gamma * self.gae_lambda * advantage * next_mask
			
			target_values[:, t, :, :] = advantage + values.data[:, t, :, :] * masks[:, t, :, :]
			
			next_value = values.data[:, t, :, :]
			next_mask = masks[:, t, :, :]

		return target_values*masks


	def nstep_returns(self, rewards, values, next_value, masks, next_mask):
		
		nstep_values = torch.zeros_like(values)
		for t_start in range(rewards.size(1)):
			nstep_return_t = torch.zeros_like(values[:, 0])
			for step in range(self.n_steps + 1):
				t = t_start + step
				if t >= rewards.size(1):
					break
				elif step == self.n_steps:
					nstep_return_t += self.gamma ** (step) * values[:, t] * masks[:, t]
				elif t == rewards.size(1) - 1: # and self.args.add_value_last_step:
					nstep_return_t += self.gamma ** (step) * rewards[:, t] * masks[:, t]
					nstep_return_t += self.gamma ** (step + 1) * next_value * next_mask 
				else:
					nstep_return_t += self.gamma ** (step) * rewards[:, t] * masks[:, t]
			nstep_values[:, t_start, :, :] = nstep_return_t
		
		return nstep_values
