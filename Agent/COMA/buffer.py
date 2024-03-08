import random
import numpy as np
import torch


class RolloutBuffer:
	def __init__(
		self, 
		num_episodes, 
		max_time_steps, 
		num_agents, 
		critic_obs_shape, 
		actor_obs_shape, 
		critic_rnn_num_layers,
		actor_rnn_num_layers,
		critic_rnn_hidden_state_dim,
		actor_rnn_hidden_state_dim,
		data_chunk_length,
		num_actions,
		lambda_,
		gamma,
		):
		self.data_chunk_length = data_chunk_length
		self.critic_rnn_num_layers = critic_rnn_num_layers
		self.critic_rnn_hidden_state_dim = critic_rnn_hidden_state_dim
		self.actor_rnn_num_layers = actor_rnn_num_layers
		self.actor_rnn_hidden_state_dim = actor_rnn_hidden_state_dim
		self.num_episodes = num_episodes
		self.max_time_steps = max_time_steps
		self.num_agents = num_agents
		self.critic_obs_shape = critic_obs_shape
		self.actor_obs_shape = actor_obs_shape
		self.num_actions = num_actions
		self.episode_num = 0
		self.time_step = 0

		self.lambda_ = lambda_
		self.gamma = gamma

		self.critic_states = np.zeros((num_episodes, max_time_steps, num_agents, critic_obs_shape))
		self.actor_states = np.zeros((num_episodes, max_time_steps, num_agents, actor_obs_shape))
		self.critic_rnn_hidden_state = np.zeros((num_episodes, max_time_steps, critic_rnn_num_layers, num_agents, critic_rnn_hidden_state_dim))
		self.actor_rnn_hidden_state = np.zeros((num_episodes, max_time_steps, actor_rnn_num_layers, num_agents, actor_rnn_hidden_state_dim))
		self.Q_values = np.zeros((num_episodes, max_time_steps, num_agents, num_actions))
		self.last_one_hot_actions = np.zeros((num_episodes, max_time_steps, num_agents, num_actions))
		self.probs = np.zeros((num_episodes, max_time_steps, num_agents, num_actions))
		self.actions = np.zeros((num_episodes, max_time_steps, num_agents), dtype=int)
		self.one_hot_actions = np.zeros((num_episodes, max_time_steps, num_agents, num_actions))
		self.action_masks = np.zeros((num_episodes, max_time_steps, num_agents, num_actions))
		self.rewards = np.zeros((num_episodes, max_time_steps))
		self.dones = np.zeros((num_episodes, max_time_steps, num_agents))
	

	def clear(self):
		self.critic_states = np.zeros((self.num_episodes, self.max_time_steps, self.num_agents, self.critic_obs_shape))
		self.actor_states = np.zeros((self.num_episodes, self.max_time_steps, self.num_agents, self.actor_obs_shape))
		self.critic_rnn_hidden_state = np.zeros((self.num_episodes, self.max_time_steps, self.critic_rnn_num_layers, self.num_agents, self.critic_rnn_hidden_state_dim))
		self.actor_rnn_hidden_state = np.zeros((self.num_episodes, self.max_time_steps, self.actor_rnn_num_layers, self.num_agents, self.actor_rnn_hidden_state_dim))
		self.Q_values = np.zeros((self.num_episodes, self.max_time_steps, self.num_agents, self.num_actions))
		self.last_one_hot_actions = np.zeros((self.num_episodes, self.max_time_steps, self.num_agents, self.num_actions))
		self.probs = np.zeros((self.num_episodes, self.max_time_steps, self.num_agents, self.num_actions))
		self.actions = np.zeros((self.num_episodes, self.max_time_steps, self.num_agents), dtype=int)
		self.one_hot_actions = np.zeros((self.num_episodes, self.max_time_steps, self.num_agents, self.num_actions))
		self.action_masks = np.zeros((self.num_episodes, self.max_time_steps, self.num_agents, self.num_actions))
		self.rewards = np.zeros((self.num_episodes, self.max_time_steps))
		self.dones = np.zeros((self.num_episodes, self.max_time_steps, self.num_agents))

		self.time_step = 0
		self.episode_num = 0

	def push(
		self, 
		critic_state, 
		actor_state, 
		critic_rnn_hidden_state,
		actor_rnn_hidden_state,
		Q_values,
		last_one_hot_actions, 
		probs,
		actions, 
		one_hot_actions, 
		action_masks, 
		rewards, 
		dones
		):

		self.critic_states[self.episode_num][self.time_step] = critic_state
		self.actor_states[self.episode_num][self.time_step] = actor_state
		self.critic_rnn_hidden_state[self.episode_num][self.time_step] = critic_rnn_hidden_state
		self.actor_rnn_hidden_state[self.episode_num][self.time_step] = actor_rnn_hidden_state
		self.Q_values[self.episode_num][self.time_step] = Q_values
		self.last_one_hot_actions[self.episode_num][self.time_step] = last_one_hot_actions
		self.probs[self.episode_num][self.time_step] = probs
		self.actions[self.episode_num][self.time_step] = actions
		self.one_hot_actions[self.episode_num][self.time_step] = one_hot_actions
		self.action_masks[self.episode_num][self.time_step] = action_masks
		self.rewards[self.episode_num][self.time_step] = rewards
		self.dones[self.episode_num][self.time_step] = dones

		if self.time_step < self.max_time_steps-1:
			self.time_step += 1


	def build_td_lambda_targets(self):
		# Assumes  <target_qs > in B*T*A and <reward >, <terminated >  in B*T*A, <mask > in (at least) B*T-1*1
		# Initialise  last  lambda -return  for  not  terminated  episodes
		Q_values = (torch.from_numpy(self.Q_values) * torch.from_numpy(self.one_hot_actions)).sum(dim=-1)
		ret = Q_values.new_zeros(*Q_values.shape)
		ret = Q_values * (1-torch.from_numpy(self.dones))
		# ret[:, -1] = target_qs[:, -1] * (1 - (torch.sum(terminated, dim=1)>0).int())
		# Backwards  recursive  update  of the "forward  view"
		for t in range(ret.shape[1] - 2, -1,  -1):
			ret[:, t] = self.lambda_ * self.gamma * ret[:, t+1] + (1-torch.from_numpy(self.dones[:, t])) \
						* (torch.from_numpy(self.rewards[:, t]).unsqueeze(-1) + (1 - self.lambda_) * self.gamma * Q_values[:, t+1] * (1 - torch.from_numpy(self.dones[:, t+1])))
		# Returns lambda-return from t=0 to t=T-1, i.e. in B*T-1*A
		# return ret[:, 0:-1]
		return ret


	def sample_recurrent_policy(self):

		data_chunks = self.max_time_steps // self.data_chunk_length
		rand_batch = np.random.permutation(self.num_episodes)
		rand_time = np.random.permutation(data_chunks)

		self.target_q_values = self.build_td_lambda_targets()
		self.advantage = torch.from_numpy(self.Q_values*self.one_hot_actions).sum(dim=-1) - torch.from_numpy(self.Q_values*self.probs).sum(dim=-1)

		critic_states = torch.from_numpy(self.critic_states).float().reshape(self.num_episodes, data_chunks, self.data_chunk_length, self.num_agents, self.critic_obs_shape)[:, rand_time][rand_batch, :].reshape(-1, self.data_chunk_length, self.num_agents, self.critic_obs_shape)
		critic_rnn_hidden_state = torch.from_numpy(self.critic_rnn_hidden_state).float().reshape(self.num_episodes, data_chunks, self.data_chunk_length, self.critic_rnn_num_layers, self.num_agents, self.critic_rnn_hidden_state_dim)[:, rand_time][rand_batch, :][:, :, 0, :, :, :].permute(2, 0, 1, 3, 4).reshape(self.critic_rnn_num_layers, -1, self.critic_rnn_hidden_state_dim)
		actor_states = torch.from_numpy(self.actor_states).float().reshape(self.num_episodes, data_chunks, self.data_chunk_length, self.num_agents, self.actor_obs_shape)[:, rand_time][rand_batch, :].reshape(-1, self.data_chunk_length, self.num_agents, self.actor_obs_shape).reshape(-1, self.data_chunk_length, self.num_agents, self.actor_obs_shape)
		actor_rnn_hidden_state = torch.from_numpy(self.actor_rnn_hidden_state).float().reshape(self.num_episodes, data_chunks, self.data_chunk_length, self.actor_rnn_num_layers, self.num_agents, self.actor_rnn_hidden_state_dim)[:, rand_time][rand_batch, :][:, :, 0, :, :, :].permute(2, 0, 1, 3, 4).reshape(self.actor_rnn_num_layers, -1, self.actor_rnn_hidden_state_dim)
		actions = torch.from_numpy(self.actions).long().reshape(self.num_episodes, data_chunks, self.data_chunk_length, self.num_agents)[:, rand_time][rand_batch, :].reshape(-1, self.data_chunk_length, self.num_agents)
		last_one_hot_actions = torch.from_numpy(self.last_one_hot_actions).float().reshape(self.num_episodes, data_chunks, self.data_chunk_length, self.num_agents, self.num_actions)[:, rand_time][rand_batch, :].reshape(-1, self.data_chunk_length, self.num_agents, self.num_actions)
		one_hot_actions = torch.from_numpy(self.one_hot_actions).float().reshape(self.num_episodes, data_chunks, self.data_chunk_length, self.num_agents, self.num_actions)[:, rand_time][rand_batch, :].reshape(-1, self.data_chunk_length, self.num_agents, self.num_actions)
		action_masks = torch.from_numpy(self.action_masks).bool().reshape(self.num_episodes, data_chunks, self.data_chunk_length, self.num_agents, self.num_actions)[:, rand_time][rand_batch, :].reshape(-1, self.data_chunk_length, self.num_agents, self.num_actions)
		masks = 1-torch.from_numpy(self.dones).float().reshape(self.num_episodes, data_chunks, self.data_chunk_length, self.num_agents)[:, rand_time][rand_batch, :].reshape(-1, self.data_chunk_length, self.num_agents)
		
		target_q_values = self.target_q_values.float().reshape(self.num_episodes, data_chunks, self.data_chunk_length, self.num_agents)[:, rand_time][rand_batch, :].reshape(-1, self.data_chunk_length, self.num_agents)
		advantage = self.advantage.float().reshape(self.num_episodes, data_chunks, self.data_chunk_length, self.num_agents)[:, rand_time][rand_batch, :].reshape(-1, self.data_chunk_length, self.num_agents)
		
		return critic_states, critic_rnn_hidden_state, actor_states, actor_rnn_hidden_state, \
		actions, last_one_hot_actions, one_hot_actions, action_masks, masks, target_q_values, advantage
