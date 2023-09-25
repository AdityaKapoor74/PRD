import torch
from torch import Tensor
import numpy as np


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
	# print(y_soft)

	if hard:
		# Straight through.
		indices = (y_soft > threshold).nonzero(as_tuple=True)
		y_hard = torch.zeros_like(logits, memory_format=torch.legacy_contiguous_format)
		y_hard[indices[0], indices[1], indices[2], indices[3]] = 1.0
		ret = y_hard - y_soft.detach() + y_soft
	else:
		# Reparametrization trick.
		ret = y_soft

	# print("GUMBEL SIGMOID")
	# print(ret)
	
	return ret


class RolloutBuffer:
	def __init__(self):
		self.states_actor = []
		self.logprobs = []
		self.actions = []
		self.one_hot_actions = []


		self.rewards = []
		self.dones = []

		
		self.states_critic = []
		# self.history_states_critic = []
		# self.Q_values = []
		# self.Values = []
	

	def clear(self):
		del self.actions[:]
		del self.states_critic[:]
		# del self.history_states_critic[:]
		# del self.Q_values[:]
		# del self.Values[:]
		del self.states_actor[:]
		del self.one_hot_actions[:]
		del self.logprobs[:]
		del self.rewards[:]
		del self.dones[:]


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
		num_actions, 
		):
		# num_episodes = 10
		self.num_episodes = num_episodes
		self.max_time_steps = max_time_steps
		self.num_agents = num_agents
		self.num_enemies = num_enemies
		self.obs_shape_critic_ally = obs_shape_critic_ally
		self.obs_shape_critic_enemy = obs_shape_critic_enemy
		self.obs_shape_actor = obs_shape_actor
		self.num_actions = num_actions
		self.episode_num = 0
		self.time_step = 0

		self.states_critic_allies = np.zeros((num_episodes, max_time_steps+1, num_agents, obs_shape_critic_ally))
		self.states_critic_enemies = np.zeros((num_episodes, max_time_steps+1, num_enemies, obs_shape_critic_enemy))
		# self.Q_values = np.zeros((num_episodes, max_time_steps, num_agents))
		# self.V_values = np.zeros((num_episodes, max_time_steps, num_agents))
		# self.weights_prd = np.zeros((num_episodes, max_time_steps, num_agents, num_agents))
		self.states_actor = np.zeros((num_episodes, max_time_steps, num_agents, obs_shape_actor))
		self.logprobs = np.zeros((num_episodes, max_time_steps, num_agents))
		self.actions = np.zeros((num_episodes, max_time_steps, num_agents), dtype=int)
		self.last_one_hot_actions = np.zeros((num_episodes, max_time_steps, num_agents, num_actions))
		self.one_hot_actions = np.zeros((num_episodes, max_time_steps+1, num_agents, num_actions))
		self.action_masks = np.zeros((num_episodes, max_time_steps, num_agents, num_actions))
		self.rewards = np.zeros((num_episodes, max_time_steps, num_agents))
		self.dones = np.ones((num_episodes, max_time_steps+1, num_agents))
		# self.masks = np.zeros((num_episodes, max_time_steps))

		self.episode_length = np.zeros(num_episodes)
	

	def clear(self):
		self.states_critic_allies = np.zeros((self.num_episodes, self.max_time_steps+1, self.num_agents, self.obs_shape_critic_ally))
		self.states_critic_enemies = np.zeros((self.num_episodes, self.max_time_steps+1, self.num_enemies, self.obs_shape_critic_enemy))
		# self.Q_values = np.zeros((self.num_episodes, self.max_time_steps, self.num_agents))
		# self.V_values = np.zeros((self.num_episodes, self.max_time_steps, self.num_agents))
		# self.weights_prd = np.zeros((self.num_episodes, self.max_time_steps, self.num_agents, self.num_agents))
		self.states_actor = np.zeros((self.num_episodes, self.max_time_steps, self.num_agents, self.obs_shape_actor))
		self.logprobs = np.zeros((self.num_episodes, self.max_time_steps, self.num_agents))
		self.actions = np.zeros((self.num_episodes, self.max_time_steps, self.num_agents), dtype=int)
		self.last_one_hot_actions = np.zeros((self.num_episodes, self.max_time_steps, self.num_agents, self.num_actions))
		self.one_hot_actions = np.zeros((self.num_episodes, self.max_time_steps+1, self.num_agents, self.num_actions))
		self.action_masks = np.zeros((self.num_episodes, self.max_time_steps, self.num_agents, self.num_actions))
		self.rewards = np.zeros((self.num_episodes, self.max_time_steps, self.num_agents))
		self.dones = np.ones((self.num_episodes, self.max_time_steps+1, self.num_agents))
		# self.masks = np.zeros((self.num_episodes, self.max_time_steps))

		self.episode_length = np.zeros(self.num_episodes)

		self.time_step = 0
		self.episode_num = 0

	def push(self, state_critic_allies, state_critic_enemies, state_actor, logprobs, actions, last_one_hot_actions, one_hot_actions, action_masks, rewards, dones):

		self.states_critic_allies[self.episode_num][self.time_step] = state_critic_allies
		self.states_critic_enemies[self.episode_num][self.time_step] = state_critic_enemies
		# self.Q_values[self.episode_num][self.time_step] = Q_value
		# self.V_values[self.episode_num][self.time_step] = V_value
		# self.weights_prd[self.episode_num][self.time_step] = weights_prd
		self.states_actor[self.episode_num][self.time_step] = state_actor
		self.logprobs[self.episode_num][self.time_step] = logprobs
		self.actions[self.episode_num][self.time_step] = actions
		self.last_one_hot_actions[self.episode_num][self.time_step] = last_one_hot_actions
		self.one_hot_actions[self.episode_num][self.time_step] = one_hot_actions
		self.action_masks[self.episode_num][self.time_step] = action_masks
		self.rewards[self.episode_num][self.time_step] = rewards
		self.dones[self.episode_num][self.time_step] = dones
		# self.masks[self.episode_num][self.time_step] = 1.

		if self.time_step < self.max_time_steps-1:
			self.time_step += 1
		# else:
		# 	self.episode_num += 1
		# 	self.time_step = 0

	def end_episode(self, t, state_critic_allies, state_critic_enemies, one_hot_actions, dones):
		self.states_critic_allies[self.episode_num][self.time_step+1] = state_critic_allies
		self.states_critic_enemies[self.episode_num][self.time_step+1] = state_critic_enemies
		self.one_hot_actions[self.episode_num][self.time_step+1] = one_hot_actions
		self.dones[self.episode_num][self.time_step+1] = dones

		self.episode_length[self.episode_num] = t
		self.episode_num += 1
		self.time_step = 0


if __name__ == '__main__':

	a = torch.tensor([[[[10.0], [-10.0]]]])

	print(gumbel_sigmoid(a, hard=True))
