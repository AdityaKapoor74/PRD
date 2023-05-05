import random
import numpy as np


class RolloutBuffer:
	def __init__(self, num_episodes, max_time_steps, num_agents, height, width, channel, num_actions):
		self.num_episodes = num_episodes
		self.max_time_steps = max_time_steps
		self.num_agents = num_agents
		self.height = height
		self.width = width
		self.channel = channel
		self.num_actions = num_actions
		self.episode_num = 0
		self.time_step = 0

		self.states = np.zeros((num_episodes, max_time_steps, num_agents, height, width, channel))
		self.last_one_hot_actions = np.zeros((num_episodes, max_time_steps, num_agents, num_actions))
		self.actions = np.zeros((num_episodes, max_time_steps, num_agents), dtype=int)
		self.one_hot_actions = np.zeros((num_episodes, max_time_steps, num_agents, num_actions))
		self.rewards = np.zeros((num_episodes, max_time_steps))
		self.dones = np.zeros((num_episodes, max_time_steps))
		self.masks = np.zeros((num_episodes, max_time_steps))
	

	def clear(self):
		self.states = np.zeros((self.num_episodes, self.max_time_steps, self.num_agents, self.height, self.width, self.channel))
		self.last_one_hot_actions = np.zeros((self.num_episodes, self.max_time_steps, self.num_agents, self.num_actions))
		self.actions = np.zeros((self.num_episodes, self.max_time_steps, self.num_agents), dtype=int)
		self.one_hot_actions = np.zeros((self.num_episodes, self.max_time_steps, self.num_agents, self.num_actions))
		self.rewards = np.zeros((self.num_episodes, self.max_time_steps))
		self.dones = np.zeros((self.num_episodes, self.max_time_steps))
		self.masks = np.zeros((self.num_episodes, self.max_time_steps))

		self.time_step = 0
		self.episode_num = 0

	def push(self, state, last_one_hot_actions, actions, one_hot_actions, rewards, dones):

		self.states[self.episode_num][self.time_step] = state
		self.last_one_hot_actions[self.episode_num][self.time_step] = last_one_hot_actions
		self.actions[self.episode_num][self.time_step] = actions
		self.one_hot_actions[self.episode_num][self.time_step] = one_hot_actions
		self.rewards[self.episode_num][self.time_step] = rewards
		self.dones[self.episode_num][self.time_step] = dones
		self.masks[self.episode_num][self.time_step] = 1.

		if self.time_step < self.max_time_steps-1:
			self.time_step += 1
		else:
			self.episode_num += 1
			self.time_step = 0