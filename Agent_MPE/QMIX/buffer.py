import random
import numpy as np

class ReplayMemory:
	def __init__(self, capacity, max_episode_len, num_agents, obs_shape, action_shape):
		self.capacity = capacity
		self.length = 0
		self.episode = 0
		self.t = 0
		self.max_episode_len = max_episode_len
		self.num_agents = num_agents
		self.obs_shape = obs_shape
		self.action_shape = action_shape
		self.buffer = dict()
		self.buffer['state'] = np.zeros((capacity, self.max_episode_len+1, num_agents, obs_shape), dtype=np.float32)
		self.buffer['actions'] = np.zeros((capacity, self.max_episode_len, num_agents), dtype=np.float32)
		self.buffer['last_one_hot_actions'] = np.zeros((capacity, self.max_episode_len+1, num_agents, action_shape), dtype=np.float32)
		self.buffer['reward'] = np.zeros((capacity, self.max_episode_len), dtype=np.float32)
		self.buffer['done'] = np.zeros((capacity, self.max_episode_len), dtype=np.float32)

		self.episode_len = np.zeros(self.capacity)

	# push once per step
	def push(self, state, actions, last_one_hot_actions, reward, done):
		self.buffer['state'][self.episode][self.t] = state
		self.buffer['actions'][self.episode][self.t] = actions
		self.buffer['last_one_hot_actions'][self.episode][self.t+1] = last_one_hot_actions
		self.buffer['reward'][self.episode][self.t] = reward
		self.buffer['done'][self.episode][self.t] = done
		self.t += 1

	def end_episode(self, state):
		self.buffer['state'][self.episode][self.t] = state
		self.episode_len[self.episode] = self.t
		if self.length < self.capacity:
			self.length += 1
		self.episode = (self.episode + 1) % self.capacity
		self.t = 0
		# clear previous data
		self.buffer['state'][self.episode] = np.zeros((self.max_episode_len+1, self.num_agents, self.obs_shape), dtype=np.float32)
		self.buffer['actions'][self.episode] = np.zeros((self.max_episode_len, self.num_agents), dtype=np.float32)
		self.buffer['last_one_hot_actions'][self.episode] = np.zeros((self.max_episode_len+1, self.num_agents, self.action_shape), dtype=np.float32)
		self.buffer['reward'][self.episode] = np.zeros((self.max_episode_len,), dtype=np.float32)
		self.buffer['done'][self.episode] = np.zeros((self.max_episode_len,), dtype=np.float32)

	def sample(self, num_episodes):
		assert num_episodes <= self.length
		batch_indices = np.random.choice(self.length, size=num_episodes, replace=False)
		state_batch = np.take(self.buffer['state'], batch_indices, axis=0)[:, :self.max_episode_len+1]
		actions_batch = np.take(self.buffer['actions'], batch_indices, axis=0)[:, :self.max_episode_len]
		last_one_hot_actions_batch = np.take(self.buffer['last_one_hot_actions'], batch_indices, axis=0)[:, :self.max_episode_len+1]
		reward_batch = np.take(self.buffer['reward'], batch_indices, axis=0)[:, :self.max_episode_len]
		done_batch = np.take(self.buffer['done'], batch_indices, axis=0)[:, :self.max_episode_len]

		max_episode_len = int(np.max(self.episode_len[batch_indices]))

		return state_batch, actions_batch, last_one_hot_actions_batch, reward_batch, done_batch, max_episode_len

	def __len__(self):
		return self.length