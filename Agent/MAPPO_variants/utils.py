import torch
from torch import Tensor
import numpy as np

from model import PopArt


def gumbel_sigmoid(logits: Tensor, tau: float = 1, hard: bool = False, threshold: float = 0.5) -> Tensor:
	"""
	Samples from the Gumbel-Sigmoid distribution and optionally discretizes.
	The discretization converts the values greater than `threshold` to 1 and the rest to 0.
	The code is adapted from the official PyTorch implementation of gumbel_softmax:
	https://pytorch.org/docs/stable/_modules/torch/nn/functional.html#gumbel_softmax
	Args:
	  logits: `[..., num_features]` unnormalized log probabilities
	  tau: non-negative scalar temperature
	  hard: if ``True``, the returned samples will be discretized,
			but will be differentiated as if it is the soft sample in autograd
	 threshold: threshold for the discretization,
				values greater than this will be set to 1 and the rest to 0
	Returns:
	  Sampled tensor of same shape as `logits` from the Gumbel-Sigmoid distribution.
	  If ``hard=True``, the returned samples are descretized according to `threshold`, otherwise they will
	  be probability distributions.
	"""
	gumbels = (
		-torch.empty_like(logits, memory_format=torch.legacy_contiguous_format).exponential_().log()
	)  # ~Gumbel(0, 1)
	gumbels = (logits + gumbels) / tau  # ~Gumbel(logits, tau)
	y_soft = gumbels.sigmoid()

	if hard:
		# Straight through.
		indices = (y_soft > threshold).nonzero(as_tuple=True)
		y_hard = torch.zeros_like(logits, memory_format=torch.legacy_contiguous_format)
		y_hard[indices[0], indices[1], indices[2], indices[3]] = 1.0
		ret = y_hard - y_soft.detach() + y_soft
	else:
		# Reparametrization trick.
		ret = y_soft
	return ret

class RolloutBuffer:
	def __init__(
		self, 
		environment,
		experiment_type,
		num_episodes, 
		max_time_steps, 
		num_agents, 
		num_enemies,
		obs_shape_critic_ally, 
		obs_shape_critic_enemy, 
		obs_shape_actor, 
		rnn_num_layers_actor,
		actor_hidden_state,
		rnn_num_layers_q,
		rnn_num_layers_v,
		q_hidden_state,
		v_hidden_state,
		num_actions, 
		transition_after,
		data_chunk_length,
		norm_returns_q,
		norm_returns_v,
		clamp_rewards,
		clamp_rewards_value_min,
		clamp_rewards_value_max,
		norm_rewards,
		target_calc_style,
		gae_lambda,
		n_steps,
		gamma,
		):
		self.environment = environment
		self.experiment_type = experiment_type
		self.num_episodes = num_episodes
		self.max_time_steps = max_time_steps
		self.num_agents = num_agents
		self.num_enemies = num_enemies
		self.obs_shape_critic_ally = obs_shape_critic_ally
		self.obs_shape_critic_enemy = obs_shape_critic_enemy
		self.obs_shape_actor = obs_shape_actor
		self.rnn_num_layers_actor = rnn_num_layers_actor
		self.actor_hidden_state = actor_hidden_state
		self.rnn_num_layers_q = rnn_num_layers_q
		self.rnn_num_layers_v = rnn_num_layers_v
		self.q_hidden_state = q_hidden_state
		self.v_hidden_state = v_hidden_state	
		self.num_actions = num_actions

		self.transition_after = transition_after
		self.data_chunk_length = data_chunk_length
		self.norm_returns_q = norm_returns_q
		self.norm_returns_v = norm_returns_v
		self.clamp_rewards = clamp_rewards
		self.clamp_rewards_value_min = clamp_rewards_value_min
		self.clamp_rewards_value_max = clamp_rewards_value_max
		self.norm_rewards = norm_rewards

		self.target_calc_style = target_calc_style
		self.gae_lambda = gae_lambda
		self.gamma = gamma
		self.n_steps = n_steps

		if self.norm_rewards:
			self.reward_norm = PopArt(input_shape=1, num_agents=self.num_agents, device=self.device)

		self.episode_num = 0
		self.time_step = 0

		self.states_critic_allies = np.zeros((num_episodes, max_time_steps, num_agents, obs_shape_critic_ally))
		self.states_critic_enemies = np.zeros((num_episodes, max_time_steps, num_enemies, obs_shape_critic_enemy))
		self.hidden_state_v = np.zeros((num_episodes, max_time_steps, rnn_num_layers_v, num_agents, v_hidden_state))
		self.Q_values = np.zeros((num_episodes, max_time_steps+1, num_agents))
		if self.experiment_type == "prd_soft_advantage_global":
			self.global_Q_values = np.zeros((num_episodes, max_time_steps+1))
			self.global_hidden_state_q = np.zeros((num_episodes, max_time_steps, rnn_num_layers_q, 1, q_hidden_state))
		self.hidden_state_q = np.zeros((num_episodes, max_time_steps, rnn_num_layers_q, num_agents, q_hidden_state))
		self.V_values = np.zeros((num_episodes, max_time_steps+1, num_agents))
		self.weights_prd = np.zeros((num_episodes, max_time_steps, num_agents, num_agents))
		self.states_actor = np.zeros((num_episodes, max_time_steps, num_agents, obs_shape_actor))
		self.hidden_state_actor = np.zeros((num_episodes, max_time_steps, rnn_num_layers_actor, num_agents, actor_hidden_state))
		self.logprobs = np.zeros((num_episodes, max_time_steps, num_agents))
		self.actions = np.zeros((num_episodes, max_time_steps, num_agents), dtype=int)
		self.action_masks = np.zeros((num_episodes, max_time_steps, num_agents, num_actions))
		self.rewards = np.zeros((num_episodes, max_time_steps, num_agents))
		self.global_rewards = np.zeros((num_episodes, max_time_steps))
		self.dones = np.ones((num_episodes, max_time_steps+1, num_agents))

		self.episode_length = np.zeros(num_episodes)

		data_chunks = max_time_steps//data_chunk_length
		self.factor = torch.ones((num_episodes*data_chunks, data_chunk_length)).float()
	

	def clear(self):

		self.states_critic_allies = np.zeros((self.num_episodes, self.max_time_steps, self.num_agents, self.obs_shape_critic_ally))
		self.states_critic_enemies = np.zeros((self.num_episodes, self.max_time_steps, self.num_enemies, self.obs_shape_critic_enemy))
		self.hidden_state_v = np.zeros((self.num_episodes, self.max_time_steps, self.rnn_num_layers_v, self.num_agents, self.v_hidden_state))
		self.Q_values = np.zeros((self.num_episodes, self.max_time_steps+1, self.num_agents))
		if self.experiment_type == "prd_soft_advantage_global":
			self.global_Q_values = np.zeros((self.num_episodes, self.max_time_steps+1))
			self.global_hidden_state_q = np.zeros((self.num_episodes, self.max_time_steps, self.rnn_num_layers_q, 1, self.q_hidden_state))
		self.hidden_state_q = np.zeros((self.num_episodes, self.max_time_steps, self.rnn_num_layers_q, self.num_agents, self.q_hidden_state))
		self.V_values = np.zeros((self.num_episodes, self.max_time_steps+1, self.num_agents))
		self.weights_prd = np.zeros((self.num_episodes, self.max_time_steps, self.num_agents, self.num_agents))
		self.states_actor = np.zeros((self.num_episodes, self.max_time_steps, self.num_agents, self.obs_shape_actor))
		self.hidden_state_actor = np.zeros((self.num_episodes, self.max_time_steps, self.rnn_num_layers_actor, self.num_agents, self.actor_hidden_state))
		self.logprobs = np.zeros((self.num_episodes, self.max_time_steps, self.num_agents))
		self.actions = np.zeros((self.num_episodes, self.max_time_steps, self.num_agents), dtype=int)
		self.action_masks = np.zeros((self.num_episodes, self.max_time_steps, self.num_agents, self.num_actions))
		self.rewards = np.zeros((self.num_episodes, self.max_time_steps, self.num_agents))
		self.global_rewards = np.zeros((self.num_episodes, self.max_time_steps))
		self.dones = np.ones((self.num_episodes, self.max_time_steps+1, self.num_agents))

		self.episode_length = np.zeros(self.num_episodes)

		self.time_step = 0
		self.episode_num = 0

		data_chunks = self.max_time_steps//self.data_chunk_length
		self.factor = torch.ones((self.num_episodes*data_chunks, self.data_chunk_length)).float()

	def push(
		self, 
		state_critic_allies, 
		state_critic_enemies, 
		q_value,
		hidden_state_q,
		weights_prd,
		global_q_value,
		global_hidden_state_q,
		value, 
		hidden_state_v,
		state_actor, 
		hidden_state_actor, 
		logprobs, 
		actions, 
		action_masks, 
		rewards, 
		global_rewards,
		dones
		):

		self.states_critic_allies[self.episode_num][self.time_step] = state_critic_allies

		if "StarCraft" in self.environment:
			self.states_critic_enemies[self.episode_num][self.time_step] = state_critic_enemies

		self.V_values[self.episode_num][self.time_step] = value
		self.hidden_state_v[self.episode_num][self.time_step] = hidden_state_v
		

		if self.experiment_type == "prd_soft_advantage_global":
			self.global_Q_values[self.episode_num][self.time_step] = global_q_value
			self.global_hidden_state_q[self.episode_num][self.time_step] = global_hidden_state_q

		if "prd" in self.experiment_type:
			self.hidden_state_q[self.episode_num][self.time_step] = hidden_state_q
			self.Q_values[self.episode_num][self.time_step] = q_value
			self.weights_prd[self.episode_num][self.time_step] = weights_prd

		self.states_actor[self.episode_num][self.time_step] = state_actor
		self.hidden_state_actor[self.episode_num][self.time_step] = hidden_state_actor
		self.logprobs[self.episode_num][self.time_step] = logprobs
		self.actions[self.episode_num][self.time_step] = actions
		self.action_masks[self.episode_num][self.time_step] = action_masks
		self.rewards[self.episode_num][self.time_step] = rewards
		self.global_rewards[self.episode_num][self.time_step] = global_rewards
		self.dones[self.episode_num][self.time_step] = dones

		if self.time_step < self.max_time_steps-1:
			self.time_step += 1


	def end_episode(
		self, 
		t, 
		q_value,
		global_q_value, 
		value, 
		dones
		):
		self.V_values[self.episode_num][self.time_step+1] = value
		if self.experiment_type == "prd_soft_advantage_global": 
			self.global_Q_values[self.episode_num][self.time_step+1] = global_q_value
		if "prd" in self.experiment_type:
			self.Q_values[self.episode_num][self.time_step+1] = q_value
		self.dones[self.episode_num][self.time_step+1] = dones

		self.episode_length[self.episode_num] = t
		self.episode_num += 1
		self.time_step = 0


	def sample_recurrent_policy(self):

		data_chunks = self.max_time_steps // self.data_chunk_length
		rand_batch = np.random.permutation(self.num_episodes)
		rand_time = np.random.permutation(data_chunks)

		first_last_actions = np.zeros((self.num_episodes, 1, self.num_agents), dtype=int) + self.num_actions

		states_critic_allies = torch.from_numpy(self.states_critic_allies).float().reshape(self.num_episodes, data_chunks, self.data_chunk_length, self.num_agents, self.obs_shape_critic_ally)[:, rand_time][rand_batch, :].reshape(-1, self.data_chunk_length, self.num_agents, self.obs_shape_critic_ally)
		states_critic_enemies = torch.from_numpy(self.states_critic_enemies).float().reshape(self.num_episodes, data_chunks, self.data_chunk_length, self.num_enemies, self.obs_shape_critic_enemy)[:, rand_time][rand_batch, :].reshape(-1, self.data_chunk_length, self.num_enemies, self.obs_shape_critic_enemy)
		if self.experiment_type == "prd_soft_advantage_global":
			global_q_values = torch.from_numpy(self.global_Q_values[:, :-1]).float().reshape(self.num_episodes, data_chunks, self.data_chunk_length)[:, rand_time][rand_batch, :].reshape(-1, self.data_chunk_length)
			target_global_q_values = self.target_global_Q_values.float().reshape(self.num_episodes, data_chunks, self.data_chunk_length)[:, rand_time][rand_batch, :].reshape(-1, self.data_chunk_length)
			global_hidden_state_q = torch.from_numpy(self.global_hidden_state_q).float().reshape(self.num_episodes, data_chunks, self.data_chunk_length, self.rnn_num_layers_q, 1, self.q_hidden_state)[:, rand_time][rand_batch, :][:, :, 0, :, :, :].permute(2, 0, 1, 3, 4).reshape(self.rnn_num_layers_q, -1, self.q_hidden_state)
		else:
			global_q_values = None
			target_global_q_values = None
			global_hidden_state_q = None
		hidden_state_q = torch.from_numpy(self.hidden_state_q).float().reshape(self.num_episodes, data_chunks, self.data_chunk_length, self.rnn_num_layers_q, self.num_agents, self.q_hidden_state)[:, rand_time][rand_batch, :][:, :, 0, :, :, :].permute(2, 0, 1, 3, 4).reshape(self.rnn_num_layers_q, -1, self.q_hidden_state)
		hidden_state_v = torch.from_numpy(self.hidden_state_v).float().reshape(self.num_episodes, data_chunks, self.data_chunk_length, self.rnn_num_layers_v, self.num_agents, self.v_hidden_state)[:, rand_time][rand_batch, :][:, :, 0, :, :, :].permute(2, 0, 1, 3, 4).reshape(self.rnn_num_layers_v, -1, self.v_hidden_state)
		states_actor = torch.from_numpy(self.states_actor).float().reshape(self.num_episodes, data_chunks, self.data_chunk_length, self.num_agents, self.obs_shape_actor)[:, rand_time][rand_batch, :].reshape(-1, self.data_chunk_length, self.num_agents, self.obs_shape_actor).reshape(-1, self.data_chunk_length, self.num_agents, self.obs_shape_actor)
		hidden_state_actor = torch.from_numpy(self.hidden_state_actor).float().reshape(self.num_episodes, data_chunks, self.data_chunk_length, self.rnn_num_layers_actor, self.num_agents, self.actor_hidden_state)[:, rand_time][rand_batch, :][:, :, 0, :, :, :].permute(2, 0, 1, 3, 4).reshape(self.rnn_num_layers_actor, -1, self.actor_hidden_state)
		logprobs = torch.from_numpy(self.logprobs).float().reshape(self.num_episodes, data_chunks, self.data_chunk_length, self.num_agents)[:, rand_time][rand_batch, :].reshape(-1, self.data_chunk_length, self.num_agents)
		last_actions = torch.from_numpy(np.concatenate((first_last_actions, self.actions[:, :-1, :]), axis=1)).long().reshape(self.num_episodes, data_chunks, self.data_chunk_length, self.num_agents)[:, rand_time][rand_batch, :].reshape(-1, self.data_chunk_length, self.num_agents)
		actions = torch.from_numpy(self.actions).long().reshape(self.num_episodes, data_chunks, self.data_chunk_length, self.num_agents)[:, rand_time][rand_batch, :].reshape(-1, self.data_chunk_length, self.num_agents)
		action_masks = torch.from_numpy(self.action_masks).bool().reshape(self.num_episodes, data_chunks, self.data_chunk_length, self.num_agents, self.num_actions)[:, rand_time][rand_batch, :].reshape(-1, self.data_chunk_length, self.num_agents, self.num_actions)
		masks = 1-torch.from_numpy(self.dones[:, :-1]).float().reshape(self.num_episodes, data_chunks, self.data_chunk_length, self.num_agents)[:, rand_time][rand_batch, :].reshape(-1, self.data_chunk_length, self.num_agents)
		
		weights_prd = torch.from_numpy(self.weights_prd).float().reshape(self.num_episodes, data_chunks, self.data_chunk_length, self.num_agents, self.num_agents)[:, rand_time][rand_batch, :].reshape(-1, self.data_chunk_length, self.num_agents, self.num_agents)
		q_values = torch.from_numpy(self.Q_values[:, :-1, :]).float().reshape(self.num_episodes, data_chunks, self.data_chunk_length, self.num_agents)[:, rand_time][rand_batch, :].reshape(-1, self.data_chunk_length, self.num_agents)
		values = torch.from_numpy(self.V_values[:, :-1, :]).float().reshape(self.num_episodes, data_chunks, self.data_chunk_length, self.num_agents)[:, rand_time][rand_batch, :].reshape(-1, self.data_chunk_length, self.num_agents)
		target_q_values = self.target_q_values.float().reshape(self.num_episodes, data_chunks, self.data_chunk_length, self.num_agents)[:, rand_time][rand_batch, :].reshape(-1, self.data_chunk_length, self.num_agents)
		target_values = self.target_values.float().reshape(self.num_episodes, data_chunks, self.data_chunk_length, self.num_agents)[:, rand_time][rand_batch, :].reshape(-1, self.data_chunk_length, self.num_agents)
		advantage = self.advantage.float().reshape(self.num_episodes, data_chunks, self.data_chunk_length, self.num_agents)[:, rand_time][rand_batch, :].reshape(-1, self.data_chunk_length, self.num_agents)
		

		factor = self.factor.reshape(self.num_episodes, data_chunks, self.data_chunk_length)[:, rand_time][rand_batch, :].reshape(-1, self.data_chunk_length)

		return states_critic_allies, states_critic_enemies, hidden_state_q, global_hidden_state_q, hidden_state_v, states_actor, hidden_state_actor, logprobs, \
		last_actions, actions, action_masks, masks, values, target_values, q_values, target_q_values, global_q_values, target_global_q_values, advantage, factor, weights_prd

	def calculate_targets(self, episode, select_above_threshold, q_value_norm=None, global_q_value_norm=None, v_value_norm=None):
		
		masks = 1 - torch.from_numpy(self.dones[:, :-1, :])
		next_mask = 1 - torch.from_numpy(self.dones[:, -1, :])

		rewards = torch.from_numpy(self.rewards)
		global_rewards = torch.from_numpy(self.global_rewards)

		values = torch.from_numpy(self.V_values[:, :-1, :]) * masks
		next_values = torch.from_numpy(self.V_values[:, -1, :]) * next_mask

		if self.norm_returns_v:
			values_shape = values.shape
			values = v_value_norm.denormalize(values.view(-1)).view(values_shape) * masks.view(values_shape)

			next_values_shape = next_values.shape
			next_values = v_value_norm.denormalize(next_values.view(-1)).view(next_values_shape) * next_mask.view(next_values_shape)

		if self.clamp_rewards:
			rewards = torch.clamp(rewards, min=self.clamp_rewards_value_min, max=self.clamp_rewards_value_max)

		if self.norm_rewards:
			rewards_shape = rewards.shape
			rewards = self.reward_norm.update(rewards.view(-1).to(self.device), masks.view(-1).to(self.device)).reshape(rewards_shape)
		
		# TARGET CALC
		if self.experiment_type in ["shared", "HAPPO"]:
			if self.target_calc_style == "GAE":
				target_values = self.gae_targets(rewards, values, next_values, masks, next_mask)
			elif self.target_calc_style == "N_steps":
				target_values = self.nstep_returns(rewards, values, next_values, masks, next_mask)

			target_q_values = rewards.new_zeros(*rewards.shape)
			target_global_q_values = rewards.new_zeros(*rewards.shape).sum(dim=-1) # to collapse num_agents
		else:
			q_values = torch.from_numpy(self.Q_values[:, :-1, :]) * masks
			next_q_values = torch.from_numpy(self.Q_values[:, -1, :]) * next_mask
			weights_prd = torch.from_numpy(self.weights_prd)

			if self.norm_returns_q:
				values_shape = q_values.shape
				q_values = q_value_norm.denormalize(q_values.view(-1)).view(values_shape) * masks.view(values_shape)

				next_values_shape = next_q_values.shape
				next_q_values = q_value_norm.denormalize(next_q_values.view(-1)).view(next_values_shape) * next_mask.view(next_values_shape)

			if self.target_calc_style == "GAE":
				target_q_values = self.gae_targets(rewards, q_values, next_q_values, masks, next_mask)
			elif self.target_calc_style == "N_steps":
				target_q_values = self.nstep_returns(rewards, q_values, next_q_values, masks, next_mask)

			if self.experiment_type == "prd_soft_advantage_global":
				global_q_values = torch.from_numpy(self.global_Q_values[:, :-1]) * (masks.sum(dim=-1)>0).float()
				next_global_q_values = torch.from_numpy(self.global_Q_values[:, -1]) * (next_mask.sum(dim=-1)>0).float()

				if self.norm_returns_q:
					values_shape = global_q_values.shape
					global_q_values = global_q_value_norm.denormalize(global_q_values.view(-1)).view(values_shape) * (masks.sum(dim=-1)>0).float().view(values_shape)

					next_values_shape = next_global_q_values.shape
					next_global_q_values = global_q_value_norm.denormalize(next_global_q_values.view(-1)).view(next_values_shape) * (next_mask.sum(dim=-1)>0).view(next_values_shape)

				if self.target_calc_style == "GAE":
					target_global_q_values = self.gae_targets(global_rewards.unsqueeze(-1), global_q_values.unsqueeze(-1), next_global_q_values.unsqueeze(-1), (masks.sum(dim=-1)>0).float().unsqueeze(-1), (next_mask.sum(dim=-1)>0).unsqueeze(-1)).squeeze(-1)
				elif self.target_calc_style == "N_steps":
					target_global_q_values = self.nstep_returns(global_rewards.unsqueeze(-1), global_q_values.unsqueeze(-1), next_global_q_values.unsqueeze(-1), (masks.sum(dim=-1)>0).float().unsqueeze(-1), (next_mask.sum(dim=-1)>0).unsqueeze(-1)).squeeze(-1)
			else:
				target_global_q_values = rewards.new_zeros(*rewards.shape).sum(dim=-1) # to collapse num_agents
			
			if self.experiment_type in ["prd_above_threshold_ascend", "prd_above_threshold_decay"]:
				mask_rewards = (weights_prd>select_above_threshold).int()
				rewards_ = (rewards.unsqueeze(2).repeat(1, 1, self.num_agents, 1) * mask_rewards).sum(dim=-1)
			elif self.experiment_type == "prd_above_threshold":
				if episode > self.transition_after:
					mask_rewards = (weights_prd>select_above_threshold).int()
					rewards_ = (rewards.unsqueeze(2).repeat(1, 1, self.num_agents, 1) * mask_rewards).sum(dim=-1)
				else:
					rewards_ = (rewards.unsqueeze(2).repeat(1, 1, self.num_agents, 1)).sum(dim=-1)
			elif "top" in self.experiment_type:
				if episode > self.transition_after:
					_, indices = torch.topk(weights_prd, k=self.top_k, dim=-1)
					mask_rewards = torch.sum(F.one_hot(indices, num_classes=self.num_agents), dim=-2)
					rewards_ = (rewards.unsqueeze(2).repeat(1, 1, self.num_agents, 1) * mask_rewards).sum(dim=-1)
				else:
					rewards_ = (rewards.unsqueeze(2).repeat(1, 1, self.num_agents, 1)).sum(dim=-1)
			elif "prd_soft_advantage" in self.experiment_type:
				if episode > self.transition_after:
					rewards_ = (rewards.unsqueeze(2).repeat(1, 1, self.num_agents, 1) * weights_prd).sum(dim=-1)
				else:
					rewards_ = (rewards.unsqueeze(2).repeat(1, 1, self.num_agents, 1)).sum(dim=-1)

			if self.target_calc_style == "GAE":
				target_values = self.gae_targets(rewards_, values, next_values, masks, next_mask)
			elif self.target_calc_style == "N_steps":
				target_values = self.nstep_returns(rewards_, values, next_values, masks, next_mask)
			

		self.advantage = (target_values - values).detach()

		self.target_q_values = target_q_values
		self.target_global_Q_values = target_global_q_values
		self.target_values = target_values


	def gae_targets(self, rewards, values, next_value, masks, next_mask):
		
		target_values = rewards.new_zeros(*rewards.shape)
		advantage = 0

		for t in reversed(range(0, rewards.shape[1])):

			td_error = rewards[:,t,:] + (self.gamma * next_value * next_mask) - values.data[:,t,:] * masks[:, t, :]
			advantage = td_error + self.gamma * self.gae_lambda * advantage * next_mask
			
			target_values[:, t, :] = advantage + values.data[:, t, :] * masks[:, t, :]
			
			next_value = values.data[:, t, :]
			next_mask = masks[:, t, :]

		return target_values * masks


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
			nstep_values[:, t_start, :] = nstep_return_t
		
		return nstep_values
