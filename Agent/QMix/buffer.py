import random
import numpy as np
import torch

class ReplayMemory:
	def __init__(self, capacity, max_episode_len, num_agents, q_obs_shape, q_mix_obs_shape, rnn_num_layers, rnn_hidden_state_shape, data_chunk_length, action_shape):
		self.capacity = capacity
		self.length = 0
		self.episode = 0
		self.t = 0
		self.max_episode_len = max_episode_len
		self.num_agents = num_agents
		self.q_obs_shape = q_obs_shape
		self.q_mix_obs_shape = q_mix_obs_shape
		self.rnn_num_layers = rnn_num_layers
		self.rnn_hidden_state_shape = rnn_hidden_state_shape
		self.data_chunk_length = data_chunk_length
		self.action_shape = action_shape
		self.buffer = dict()
		self.buffer['state'] = np.zeros((capacity, self.max_episode_len, num_agents, q_obs_shape), dtype=np.float32)
		self.buffer['rnn_hidden_state'] = np.zeros((capacity, self.max_episode_len, rnn_num_layers, num_agents, rnn_hidden_state_shape), dtype=np.float32)
		self.buffer['next_state'] = np.zeros((capacity, self.max_episode_len, num_agents, q_obs_shape), dtype=np.float32)
		self.buffer['next_rnn_hidden_state'] = np.zeros((capacity, self.max_episode_len, rnn_num_layers, num_agents, rnn_hidden_state_shape), dtype=np.float32)
		self.buffer['full_state'] = np.zeros((capacity, self.max_episode_len, 1, q_mix_obs_shape), dtype=np.float32)
		self.buffer['next_full_state'] = np.zeros((capacity, self.max_episode_len, 1, q_mix_obs_shape), dtype=np.float32)
		self.buffer['actions'] = np.zeros((capacity, self.max_episode_len, num_agents), dtype=np.float32)
		self.buffer['last_one_hot_actions'] = np.zeros((capacity, self.max_episode_len, num_agents, action_shape), dtype=np.float32)
		self.buffer['next_last_one_hot_actions'] = np.zeros((capacity, self.max_episode_len, num_agents, action_shape), dtype=np.float32)
		self.buffer['next_mask_actions'] = np.zeros((capacity, self.max_episode_len, num_agents, action_shape), dtype=np.float32)
		self.buffer['reward'] = np.zeros((capacity, self.max_episode_len), dtype=np.float32)
		self.buffer['indiv_dones'] = np.ones((capacity, self.max_episode_len, num_agents), dtype=np.float32)
		self.buffer['next_indiv_dones'] = np.ones((capacity, self.max_episode_len, num_agents), dtype=np.float32)
		self.buffer['done'] = np.ones((capacity, self.max_episode_len), dtype=np.float32)
		self.buffer['mask'] = np.zeros((capacity, self.max_episode_len), dtype=np.float32)

		self.episode_len = np.zeros(self.capacity)

	# push once per step
	def push(self, state, rnn_hidden_state, full_state, actions, last_one_hot_actions, next_state, next_rnn_hidden_state, next_full_state, next_last_one_hot_actions, next_mask_actions, reward, done, indiv_dones, next_indiv_dones):
		self.buffer['state'][self.episode][self.t] = state
		self.buffer['rnn_hidden_state'][self.episode][self.t] = rnn_hidden_state
		self.buffer['full_state'][self.episode][self.t] = full_state
		self.buffer['actions'][self.episode][self.t] = actions
		self.buffer['last_one_hot_actions'][self.episode][self.t] = last_one_hot_actions
		self.buffer['next_state'][self.episode][self.t] = next_state
		self.buffer['next_rnn_hidden_state'][self.episode][self.t] = next_rnn_hidden_state
		self.buffer['next_full_state'][self.episode][self.t] = next_full_state
		self.buffer['next_last_one_hot_actions'][self.episode][self.t] = next_last_one_hot_actions
		self.buffer['next_mask_actions'][self.episode][self.t] = next_mask_actions
		self.buffer['reward'][self.episode][self.t] = reward
		self.buffer['done'][self.episode][self.t] = done
		self.buffer['indiv_dones'][self.episode][self.t] = indiv_dones
		self.buffer['next_indiv_dones'][self.episode][self.t] = next_indiv_dones
		self.buffer['mask'][self.episode][self.t] = 1.
		self.t += 1

	def end_episode(self):
		self.episode_len[self.episode] = self.t
		if self.length < self.capacity:
			self.length += 1
		self.episode = (self.episode + 1) % self.capacity
		self.t = 0
		# clear previous data
		self.buffer['state'][self.episode] = np.zeros((self.max_episode_len, self.num_agents, self.q_obs_shape), dtype=np.float32)
		self.buffer['rnn_hidden_state'][self.episode] = np.zeros((self.max_episode_len, self.rnn_num_layers, self.num_agents, self.rnn_hidden_state_shape), dtype=np.float32)
		self.buffer['full_state'][self.episode] = np.zeros((self.max_episode_len, 1, self.q_mix_obs_shape), dtype=np.float32)
		self.buffer['actions'][self.episode] = np.zeros((self.max_episode_len, self.num_agents), dtype=np.float32)
		self.buffer['last_one_hot_actions'][self.episode] = np.zeros((self.max_episode_len, self.num_agents, self.action_shape), dtype=np.float32)
		self.buffer['next_state'][self.episode] = np.zeros((self.max_episode_len, self.num_agents, self.q_obs_shape), dtype=np.float32)
		self.buffer['next_rnn_hidden_state'][self.episode] = np.zeros((self.max_episode_len, self.rnn_num_layers, self.num_agents, self.rnn_hidden_state_shape), dtype=np.float32)
		self.buffer['next_full_state'][self.episode] = np.zeros((self.max_episode_len, 1, self.q_mix_obs_shape), dtype=np.float32)
		self.buffer['next_last_one_hot_actions'][self.episode] = np.zeros((self.max_episode_len, self.num_agents, self.action_shape), dtype=np.float32)
		self.buffer['next_mask_actions'][self.episode] = np.zeros((self.max_episode_len, self.num_agents, self.action_shape), dtype=np.float32)
		self.buffer['reward'][self.episode] = np.zeros((self.max_episode_len,), dtype=np.float32)
		self.buffer['done'][self.episode] = np.ones((self.max_episode_len,), dtype=np.float32)
		self.buffer['indiv_dones'][self.episode] = np.ones((self.max_episode_len, self.num_agents), dtype=np.float32)
		self.buffer['next_indiv_dones'][self.episode] = np.ones((self.max_episode_len, self.num_agents), dtype=np.float32)
		self.buffer['mask'][self.episode] = np.zeros((self.max_episode_len,), dtype=np.float32)

	def sample(self, num_episodes):
		assert num_episodes <= self.length

		data_chunks = self.max_episode_len // self.data_chunk_length
		batch_indices = np.random.choice(self.length, size=num_episodes, replace=False)
		rand_time = np.random.permutation(data_chunks)
		
		state_batch = torch.from_numpy(np.take(self.buffer['state'], batch_indices, axis=0)).reshape(num_episodes, data_chunks, self.data_chunk_length, self.num_agents, -1)[:, rand_time].reshape(num_episodes*data_chunks, self.data_chunk_length, self.num_agents, -1)
		rnn_hidden_state_batch = torch.from_numpy(np.take(self.buffer['rnn_hidden_state'], batch_indices, axis=0)).reshape(num_episodes, data_chunks, self.data_chunk_length, self.rnn_num_layers, self.num_agents, -1)[:, rand_time][:, :, 0].permute(2, 0, 1, 3, 4).reshape(self.rnn_num_layers, num_episodes*data_chunks*self.num_agents, -1)
		full_state_batch = torch.from_numpy(np.take(self.buffer['full_state'], batch_indices, axis=0)).reshape(num_episodes, data_chunks, self.data_chunk_length, 1, -1)[:, rand_time].reshape(num_episodes*data_chunks, self.data_chunk_length, -1)
		actions_batch = torch.from_numpy(np.take(self.buffer['actions'], batch_indices, axis=0)).long().reshape(num_episodes, data_chunks, self.data_chunk_length, self.num_agents, -1)[:, rand_time].reshape(num_episodes*data_chunks, self.data_chunk_length, self.num_agents, -1)
		last_one_hot_actions_batch = torch.from_numpy(np.take(self.buffer['last_one_hot_actions'], batch_indices, axis=0)).reshape(num_episodes, data_chunks, self.data_chunk_length, self.num_agents, -1)[:, rand_time].reshape(num_episodes*data_chunks, self.data_chunk_length, self.num_agents, -1)
		next_state_batch = torch.from_numpy(np.take(self.buffer['next_state'], batch_indices, axis=0)).reshape(num_episodes, data_chunks, self.data_chunk_length, self.num_agents, -1)[:, rand_time].reshape(num_episodes*data_chunks, self.data_chunk_length, self.num_agents, -1)
		next_rnn_hidden_state_batch = torch.from_numpy(np.take(self.buffer['next_rnn_hidden_state'], batch_indices, axis=0)).reshape(num_episodes, data_chunks, self.data_chunk_length, self.rnn_num_layers, self.num_agents, -1)[:, rand_time][:, :, 0].permute(2, 0, 1, 3, 4).reshape(self.rnn_num_layers, num_episodes*data_chunks*self.num_agents, -1)
		next_full_state_batch = torch.from_numpy(np.take(self.buffer['next_full_state'], batch_indices, axis=0)).reshape(num_episodes, data_chunks, self.data_chunk_length, 1, -1)[:, rand_time].reshape(num_episodes*data_chunks, self.data_chunk_length, -1)
		next_last_one_hot_actions_batch = torch.from_numpy(np.take(self.buffer['next_last_one_hot_actions'], batch_indices, axis=0)).reshape(num_episodes, data_chunks, self.data_chunk_length, self.num_agents, -1)[:, rand_time].reshape(num_episodes*data_chunks, self.data_chunk_length, self.num_agents, -1)
		next_mask_actions_batch = torch.from_numpy(np.take(self.buffer['next_mask_actions'], batch_indices, axis=0)).bool().reshape(num_episodes, data_chunks, self.data_chunk_length, self.num_agents, -1)[:, rand_time].reshape(num_episodes*data_chunks, self.data_chunk_length, self.num_agents, -1)
		reward_batch = torch.from_numpy(np.take(self.buffer['reward'], batch_indices, axis=0)).reshape(num_episodes, data_chunks, self.data_chunk_length, -1)[:, rand_time].reshape(num_episodes*data_chunks, self.data_chunk_length, -1)
		done_batch = torch.from_numpy(np.take(self.buffer['done'], batch_indices, axis=0)).reshape(num_episodes, data_chunks, self.data_chunk_length, -1)[:, rand_time].reshape(num_episodes*data_chunks, self.data_chunk_length, -1)
		indiv_dones_batch = torch.from_numpy(np.take(self.buffer['indiv_dones'], batch_indices, axis=0)).reshape(num_episodes, data_chunks, self.data_chunk_length, self.num_agents, -1)[:, rand_time].reshape(num_episodes*data_chunks, self.data_chunk_length, self.num_agents, -1)
		next_indiv_dones_batch = torch.from_numpy(np.take(self.buffer['next_indiv_dones'], batch_indices, axis=0)).reshape(num_episodes, data_chunks, self.data_chunk_length, self.num_agents, -1)[:, rand_time].reshape(num_episodes*data_chunks, self.data_chunk_length, self.num_agents, -1)
		team_mask_batch = torch.from_numpy(np.take(self.buffer['mask'], batch_indices, axis=0)).reshape(num_episodes, data_chunks, self.data_chunk_length, -1)[:, rand_time].reshape(num_episodes*data_chunks, self.data_chunk_length, -1)

		max_episode_len = int(np.max(self.episode_len[batch_indices]))

		return state_batch, rnn_hidden_state_batch, full_state_batch, actions_batch, last_one_hot_actions_batch, next_state_batch, next_rnn_hidden_state_batch, next_full_state_batch, \
		next_last_one_hot_actions_batch, next_mask_actions_batch, reward_batch, done_batch, indiv_dones_batch, next_indiv_dones_batch, team_mask_batch, max_episode_len

	def __len__(self):
		return self.length
