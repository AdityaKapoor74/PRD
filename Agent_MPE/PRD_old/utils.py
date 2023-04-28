import numpy as np

class RolloutBuffer:
	def __init__(self, num_episodes, max_time_steps, num_agents, obs_shape_critic, obs_shape_actor, num_actions):
		self.num_episodes = num_episodes
		self.max_time_steps = max_time_steps
		self.num_agents = num_agents
		self.obs_shape_critic = obs_shape_critic
		self.obs_shape_actor = obs_shape_actor
		self.num_actions = num_actions
		self.episode_num = 0
		self.time_step = 0

		self.states_critic = np.zeros((num_episodes, max_time_steps, num_agents, obs_shape_critic))
		self.states_actor = np.zeros((num_episodes, max_time_steps, num_agents, obs_shape_actor))
		self.actions = np.zeros((num_episodes, max_time_steps, num_agents), dtype=int)
		self.one_hot_actions = np.zeros((num_episodes, max_time_steps, num_agents, num_actions))
		self.rewards = np.zeros((num_episodes, max_time_steps, num_agents))
		self.dones = np.zeros((num_episodes, max_time_steps, num_agents))
		self.masks = np.zeros((num_episodes, max_time_steps))
	

	def clear(self):
		self.states_critic = np.zeros((self.num_episodes, self.max_time_steps, self.num_agents, self.obs_shape_critic))
		self.states_actor = np.zeros((self.num_episodes, self.max_time_steps, self.num_agents, self.obs_shape_actor))
		self.actions = np.zeros((self.num_episodes, self.max_time_steps, self.num_agents), dtype=int)
		self.one_hot_actions = np.zeros((self.num_episodes, self.max_time_steps, self.num_agents, self.num_actions))
		self.rewards = np.zeros((self.num_episodes, self.max_time_steps, self.num_agents))
		self.dones = np.zeros((self.num_episodes, self.max_time_steps, self.num_agents))
		self.masks = np.zeros((self.num_episodes, self.max_time_steps))

		self.time_step = 0
		self.episode_num = 0

	def push(self, state_critic, state_actor, actions, one_hot_actions, rewards, dones):

		self.states_critic[self.episode_num][self.time_step] = state_critic
		self.states_actor[self.episode_num][self.time_step] = state_actor
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
