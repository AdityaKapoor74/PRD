import random
import numpy as np
import torch


class RolloutBuffer:
	def __init__(
		self, num_episodes, 
		max_time_steps, 
		num_agents, 
		critic_obs_shape, 
		actor_obs_shape, 
		num_actions, 
		rnn_num_layers, 
		rnn_hidden_dim, 
		data_chunk_length,
		gamma,
		lambda_,
		device,
		):
		self.num_episodes = num_episodes
		self.max_time_steps = max_time_steps
		self.num_agents = num_agents
		self.critic_obs_shape = critic_obs_shape
		self.actor_obs_shape = actor_obs_shape
		self.num_actions = num_actions
		self.rnn_num_layers = rnn_num_layers
		self.rnn_hidden_dim = rnn_hidden_dim
		self.data_chunk_length = data_chunk_length
		self.gamma = gamma
		self.lambda_ = lambda_
		self.device = device
		self.episode_num = 0
		self.time_step = 0

		self.critic_states = np.zeros((num_episodes, max_time_steps+1, critic_obs_shape))
		self.actor_states = np.zeros((num_episodes, max_time_steps, num_agents, actor_obs_shape))
		self.actor_states = np.zeros((num_episodes, max_time_steps, num_agents, actor_obs_shape))
		self.rnn_hidden_state = np.zeros((num_episodes, max_time_steps, rnn_num_layers, num_agents, rnn_hidden_dim))
		self.actions = np.zeros((num_episodes, max_time_steps, num_agents), dtype=int)
		self.one_hot_actions = np.zeros((num_episodes, max_time_steps+1, num_agents, num_actions))
		self.mask_actions = np.zeros((num_episodes, max_time_steps, num_agents, num_actions))
		self.rewards = np.zeros((num_episodes, max_time_steps))
		self.dones = np.zeros((num_episodes, max_time_steps+1))
		self.masks = np.zeros((num_episodes, max_time_steps))
	

	def clear(self):
		self.critic_states = np.zeros((self.num_episodes, self.max_time_steps+1, self.critic_obs_shape))
		self.actor_states = np.zeros((self.num_episodes, self.max_time_steps, self.num_agents, self.actor_obs_shape))
		self.rnn_hidden_state = np.zeros((self.num_episodes, self.max_time_steps, self.rnn_num_layers, self.num_agents, self.rnn_hidden_dim))
		self.actions = np.zeros((self.num_episodes, self.max_time_steps, self.num_agents), dtype=int)
		self.one_hot_actions = np.zeros((self.num_episodes, self.max_time_steps+1, self.num_agents, self.num_actions))
		self.mask_actions = np.zeros((self.num_episodes, self.max_time_steps, self.num_agents, self.num_actions))
		self.rewards = np.zeros((self.num_episodes, self.max_time_steps))
		self.dones = np.zeros((self.num_episodes, self.max_time_steps+1))
		self.masks = np.zeros((self.num_episodes, self.max_time_steps))

		self.time_step = 0
		self.episode_num = 0

	def push(self, critic_state, actor_state, rnn_hidden_state, actions, one_hot_actions, mask_actions, rewards, dones):

		self.critic_states[self.episode_num][self.time_step] = critic_state
		self.actor_states[self.episode_num][self.time_step] = actor_state
		self.rnn_hidden_state[self.episode_num][self.time_step] = rnn_hidden_state
		self.actions[self.episode_num][self.time_step] = actions
		self.one_hot_actions[self.episode_num][self.time_step] = one_hot_actions
		self.mask_actions[self.episode_num][self.time_step] = mask_actions
		self.rewards[self.episode_num][self.time_step] = rewards
		self.dones[self.episode_num][self.time_step] = dones
		self.masks[self.episode_num][self.time_step] = 1.

		if self.time_step < self.max_time_steps-1:
			self.time_step += 1

	def end_episode(self, critic_states, one_hot_actions, dones):
		self.critic_states[self.episode_num][self.time_step+1] = critic_states
		self.one_hot_actions[self.episode_num][self.time_step+1] = one_hot_actions
		self.dones[self.episode_num][self.time_step+1] = dones

		self.episode_num += 1
		self.time_step = 0


	def build_td_lambda_targets(self, rewards, terminations, q_values, next_q_values):
		"""
		Calculate the TD(lambda) targets for a batch of episodes.
		
		:param rewards: A tensor of shape [B, T, A] containing rewards received, where B is the batch size,
						T is the time horizon, and A is the number of agents.
		:param terminations: A tensor of shape [B, T, A] indicating whether each timestep is terminal.
		:param masks: A tensor of shape [B, T-1, 1] indicating the validity of each timestep 
					  (1 for valid, 0 for invalid).
		:param q_values: A tensor of shape [B, T, A] containing the Q-values for each state-action pair.
		:param gamma: A scalar indicating the discount factor.
		:param lambda_: A scalar indicating the decay rate for mixing n-step returns.
		
		:return: A tensor of shape [B, T, A] containing the TD-lambda targets for each timestep and agent.
		"""
		# Initialize the last lambda-return for not terminated episodes
		B, T = q_values.shape
		ret = q_values.new_zeros(B, T)  # Initialize return tensor
		ret[:, -1] = (rewards[:, T-1] + next_q_values[:, T-1]) * (1 - terminations[:, -1])  # Terminal values for the last timestep

		# Backward recursive update of the TD-lambda targets
		for t in reversed(range(T-1)):
			td_error = rewards[:, t] + self.gamma * next_q_values[:, t] * (1 - terminations[:, t + 1]) - q_values[:, t]
			ret[:, t] = q_values[:, t] + td_error * (1 - terminations[:, t + 1]) + self.lambda_ * self.gamma * ret[:, t + 1] * (1 - terminations[:, t + 1])

		return ret


	def calculate_targets(self, target_critic):
		one_hot_actions_batch = torch.from_numpy(self.one_hot_actions[:, :-1]).float()
		full_state_batch = torch.from_numpy(self.critic_states[:, :-1]).float()
		next_one_hot_actions_batch = torch.from_numpy(self.one_hot_actions[:, 1:]).float()
		next_full_state_batch = torch.from_numpy(self.critic_states[:, 1:]).float()
		reward_batch = torch.from_numpy(self.rewards).float()
		done_batch = torch.from_numpy(self.dones[:, :-1]).float()

		with torch.no_grad():
			target_Qs = target_critic(one_hot_actions_batch.to(self.device), full_state_batch.to(self.device)).squeeze(-1)
			next_target_Qs = target_critic(next_one_hot_actions_batch.to(self.device), next_full_state_batch.to(self.device)).squeeze(-1)
		
		TD_target_Qs = self.build_td_lambda_targets(reward_batch.to(self.device), done_batch.to(self.device), target_Qs, next_target_Qs)
		TD_target_Qs *= (1-done_batch).to(self.device)

		self.TD_target_Qs = TD_target_Qs.cpu()


	def sample(self):
		data_chunks = self.max_time_steps // self.data_chunk_length
		rand_batch = np.random.permutation(self.num_episodes)
		rand_time = np.random.permutation(data_chunks)
		
		state_batch = torch.from_numpy(self.actor_states).float().reshape(self.num_episodes, data_chunks, self.data_chunk_length, self.num_agents, -1)[:, rand_time][rand_batch, :].reshape(self.num_episodes*data_chunks, self.data_chunk_length, self.num_agents, -1)
		rnn_hidden_state_batch = torch.from_numpy(self.rnn_hidden_state).float().reshape(self.num_episodes, data_chunks, self.data_chunk_length, self.rnn_num_layers, self.num_agents, -1)[:, rand_time][rand_batch, :][:, :, 0].permute(2, 0, 1, 3, 4).reshape(self.rnn_num_layers, self.num_episodes*data_chunks*self.num_agents, -1)
		full_state_batch = torch.from_numpy(self.critic_states[:, :-1]).float().reshape(self.num_episodes, data_chunks, self.data_chunk_length, -1)[:, rand_time][rand_batch, :].reshape(self.num_episodes*data_chunks, self.data_chunk_length, -1)
		next_full_state_batch = torch.from_numpy(self.critic_states[:, 1:]).float().reshape(self.num_episodes, data_chunks, self.data_chunk_length, -1)[:, rand_time][rand_batch, :].reshape(self.num_episodes*data_chunks, self.data_chunk_length, -1)
		actions_batch = torch.from_numpy(self.actions).float().reshape(self.num_episodes, data_chunks, self.data_chunk_length, self.num_agents, -1)[:, rand_time][rand_batch, :].reshape(self.num_episodes*data_chunks, self.data_chunk_length, self.num_agents, -1)
		one_hot_actions_batch = torch.from_numpy(self.one_hot_actions[:, :-1]).float().reshape(self.num_episodes, data_chunks, self.data_chunk_length, self.num_agents, -1)[:, rand_time][rand_batch, :].reshape(self.num_episodes*data_chunks, self.data_chunk_length, self.num_agents, -1)
		next_one_hot_actions_batch = torch.from_numpy(self.one_hot_actions[:, 1:]).float().reshape(self.num_episodes, data_chunks, self.data_chunk_length, self.num_agents, -1)[:, rand_time][rand_batch, :].reshape(self.num_episodes*data_chunks, self.data_chunk_length, self.num_agents, -1)
		action_masks_batch = torch.from_numpy(self.mask_actions).bool().reshape(self.num_episodes, data_chunks, self.data_chunk_length, self.num_agents, -1)[:, rand_time][rand_batch, :].reshape(self.num_episodes*data_chunks, self.data_chunk_length, self.num_agents, -1)
		reward_batch = torch.from_numpy(self.rewards).float().reshape(self.num_episodes, data_chunks, self.data_chunk_length, -1)[:, rand_time][rand_batch, :].reshape(self.num_episodes*data_chunks, self.data_chunk_length, -1)
		done_batch = torch.from_numpy(self.dones[:, :-1]).float().reshape(self.num_episodes, data_chunks, self.data_chunk_length, -1)[:, rand_time][rand_batch, :].reshape(self.num_episodes*data_chunks, self.data_chunk_length, -1)
		next_done_batch = torch.from_numpy(self.dones[:, 1:]).float().reshape(self.num_episodes, data_chunks, self.data_chunk_length, -1)[:, rand_time][rand_batch, :].reshape(self.num_episodes*data_chunks, self.data_chunk_length, -1)
		mask_batch = torch.from_numpy(self.masks).float().reshape(self.num_episodes, data_chunks, self.data_chunk_length, -1)[:, rand_time][rand_batch, :].reshape(self.num_episodes*data_chunks, self.data_chunk_length, -1)

		TD_target_Qs_batch = self.TD_target_Qs.reshape(self.num_episodes, data_chunks, self.data_chunk_length)[:, rand_time][rand_batch, :].reshape(self.num_episodes, data_chunks*self.data_chunk_length)


		return state_batch, rnn_hidden_state_batch, full_state_batch, actions_batch, one_hot_actions_batch, action_masks_batch, next_full_state_batch, next_one_hot_actions_batch, reward_batch, done_batch, next_done_batch, mask_batch, TD_target_Qs_batch
