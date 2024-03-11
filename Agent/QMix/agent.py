import random
import numpy as np
import torch

class ReplayMemory:
	def __init__(
		self, 
		capacity, 
		max_episode_len, 
		num_agents, 
		q_obs_shape, 
		q_mix_obs_shape, 
		rnn_num_layers, 
		rnn_hidden_state_shape, 
		data_chunk_length, 
		action_shape,
		gamma,
		lambda_,
		device,
		):
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
		self.gamma = gamma
		self.lambda_ = lambda_
		self.device = device

		self.buffer = dict()
		self.buffer['state'] = np.zeros((capacity, self.max_episode_len, num_agents, q_obs_shape), dtype=np.float32)
		self.buffer['rnn_hidden_state'] = np.zeros((capacity, self.max_episode_len, rnn_num_layers, num_agents, rnn_hidden_state_shape), dtype=np.float32)
		self.buffer['mask_actions'] = np.zeros((capacity, self.max_episode_len, num_agents, action_shape), dtype=np.float32)
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
	def push(self, state, rnn_hidden_state, full_state, actions, last_one_hot_actions, mask_actions, next_state, next_rnn_hidden_state, next_full_state, next_last_one_hot_actions, next_mask_actions, reward, done, indiv_dones, next_indiv_dones):
		self.buffer['state'][self.episode][self.t] = state
		self.buffer['rnn_hidden_state'][self.episode][self.t] = rnn_hidden_state
		self.buffer['full_state'][self.episode][self.t] = full_state
		self.buffer['actions'][self.episode][self.t] = actions
		self.buffer['last_one_hot_actions'][self.episode][self.t] = last_one_hot_actions
		self.buffer['mask_actions'][self.episode][self.t] = mask_actions
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
		self.buffer['mask_actions'][self.episode] = np.zeros((self.max_episode_len, self.num_agents, self.action_shape), dtype=np.float32)
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


	def build_td_lambda_targets(self, rewards, terminated, target_qs):
		# Assumes  <target_qs > in B*T*A and <reward >, <terminated >  in B*T*A
		# Initialise  last  lambda -return  for  not  terminated  episodes
		# print(rewards.shape, terminated.shape, mask.shape, target_qs.shape)
		ret = target_qs.new_zeros(*target_qs.shape)
		ret = target_qs * (1-terminated)
		# ret[:, -1] = target_qs[:, -1] * (1 - (torch.sum(terminated, dim=1)>0).int())
		# Backwards  recursive  update  of the "forward  view"
		for t in range(ret.shape[1] - 2, -1,  -1):
			ret[:, t] = self.lambda_ * self.gamma * ret[:, t + 1] + (1-terminated[:, t]) \
						* (rewards[:, t] + (1 - self.lambda_) * self.gamma * target_qs[:, t] * (1 - terminated[:, t]))
		# Returns lambda-return from t=0 to t=T-1, i.e. in B*T-1*A
		# return ret[:, 0:-1]
		return ret

	
	# def build_td_lambda_targets(self, rewards, terminations, q_values, next_q_values):
	# 	"""
	# 	Calculate the TD(lambda) targets for a batch of episodes.
		
	# 	:param rewards: A tensor of shape [B, T, A] containing rewards received, where B is the batch size,
	# 					T is the time horizon, and A is the number of agents.
	# 	:param terminations: A tensor of shape [B, T, A] indicating whether each timestep is terminal.
	# 	:param masks: A tensor of shape [B, T-1, 1] indicating the validity of each timestep 
	# 				  (1 for valid, 0 for invalid).
	# 	:param q_values: A tensor of shape [B, T, A] containing the Q-values for each state-action pair.
	# 	:param gamma: A scalar indicating the discount factor.
	# 	:param lambda_: A scalar indicating the decay rate for mixing n-step returns.
		
	# 	:return: A tensor of shape [B, T, A] containing the TD-lambda targets for each timestep and agent.
	# 	"""
	# 	# Initialize the last lambda-return for not terminated episodes
	# 	B, T = q_values.shape
	# 	ret = q_values.new_zeros(B, T)  # Initialize return tensor
	# 	ret[:, -1] = (rewards + next_q_values) * (1 - terminations)  # Terminal values for the last timestep but since we don't know if the max_timestep is the last timestep we do it for all and overwrite it later

	# 	# Backward recursive update of the TD-lambda targets
	# 	for t in reversed(range(T-1)):
	# 		td_error = rewards[:, t] + self.gamma * next_q_values[:, t] * (1 - terminations[:, t + 1]) - q_values[:, t]
	# 		ret[:, t] = q_values[:, t] + td_error * (1 - terminations[:, t + 1]) + self.lambda_ * self.gamma * ret[:, t + 1] * (1 - terminations[:, t + 1])

	# 	return ret





	def sample(self, num_episodes, Q_network, target_Q_network, target_QMix_network):
		assert num_episodes <= self.length

		data_chunks = self.max_episode_len // self.data_chunk_length
		batch_indices = np.random.choice(self.length, size=num_episodes, replace=False)
		rand_time = np.random.permutation(data_chunks)

		next_state_batch = torch.from_numpy(np.take(self.buffer['next_state'], batch_indices, axis=0))
		next_last_one_hot_actions_batch = torch.from_numpy(np.take(self.buffer['next_last_one_hot_actions'], batch_indices, axis=0))
		next_rnn_hidden_state_batch = torch.from_numpy(np.take(self.buffer['next_rnn_hidden_state'], batch_indices, axis=0))
		next_mask_actions_batch = torch.from_numpy(np.take(self.buffer['next_mask_actions'], batch_indices, axis=0)).bool()
		next_full_state_batch = torch.from_numpy(np.take(self.buffer['next_full_state'], batch_indices, axis=0))
		state_batch = torch.from_numpy(np.take(self.buffer['state'], batch_indices, axis=0))
		last_one_hot_actions_batch = torch.from_numpy(np.take(self.buffer['last_one_hot_actions'], batch_indices, axis=0))
		rnn_hidden_state_batch = torch.from_numpy(np.take(self.buffer['rnn_hidden_state'], batch_indices, axis=0))
		mask_actions_batch = torch.from_numpy(np.take(self.buffer['mask_actions'], batch_indices, axis=0)).bool()
		full_state_batch = torch.from_numpy(np.take(self.buffer['full_state'], batch_indices, axis=0))
		reward_batch = torch.from_numpy(np.take(self.buffer['reward'], batch_indices, axis=0))
		done_batch = torch.from_numpy(np.take(self.buffer['done'], batch_indices, axis=0))

		with torch.no_grad():
			# Calculating next Q values of MAS using target network
			next_final_state_batch = torch.cat([next_state_batch, next_last_one_hot_actions_batch], dim=-1)
			next_Q_evals, _ = Q_network(
				next_final_state_batch.reshape(num_episodes*data_chunks, self.data_chunk_length, self.num_agents, -1).to(self.device), 
				next_rnn_hidden_state_batch.reshape(num_episodes, data_chunks, self.data_chunk_length, self.rnn_num_layers, self.num_agents, -1)[:, :, 0].permute(2, 0, 1, 3, 4).reshape(self.rnn_num_layers, num_episodes*data_chunks*self.num_agents, -1).to(self.device),
				next_mask_actions_batch.reshape(num_episodes*data_chunks, self.data_chunk_length, self.num_agents, -1).to(self.device)
				)
			next_Q_target, _ = target_Q_network(
				next_final_state_batch.reshape(num_episodes*data_chunks, self.data_chunk_length, self.num_agents, -1).to(self.device), 
				next_rnn_hidden_state_batch.reshape(num_episodes, data_chunks, self.data_chunk_length, self.rnn_num_layers, self.num_agents, -1)[:, :, 0].permute(2, 0, 1, 3, 4).reshape(self.rnn_num_layers, num_episodes*data_chunks*self.num_agents, -1).to(self.device),
				next_mask_actions_batch.reshape(num_episodes*data_chunks, self.data_chunk_length, self.num_agents, -1).to(self.device)
				)
			next_a_argmax = torch.argmax(next_Q_evals, dim=-1, keepdim=True)
			next_Q_target = torch.gather(next_Q_target, dim=-1, index=next_a_argmax.to(self.device)).squeeze(-1)
			next_Q_mix_target = target_QMix_network(
			next_Q_target, 
			next_full_state_batch.reshape(num_episodes*data_chunks, self.data_chunk_length, -1).to(self.device), 
			).reshape(-1) #* team_mask_batch.reshape(-1).to(self.device)

			# Calculating current Q values of MAS using target network
			# final_state_batch = torch.cat([state_batch, last_one_hot_actions_batch], dim=-1)
			# Q_evals, _ = Q_network(
			# 	final_state_batch.reshape(num_episodes*data_chunks, self.data_chunk_length, self.num_agents, -1).to(self.device), 
			# 	rnn_hidden_state_batch.reshape(num_episodes, data_chunks, self.data_chunk_length, self.rnn_num_layers, self.num_agents, -1)[:, :, 0].permute(2, 0, 1, 3, 4).reshape(self.rnn_num_layers, num_episodes*data_chunks*self.num_agents, -1).to(self.device),
			# 	mask_actions_batch.reshape(num_episodes*data_chunks, self.data_chunk_length, self.num_agents, -1).to(self.device)
			# 	)
			# Q_target, _ = target_Q_network(
			# 	final_state_batch.reshape(num_episodes*data_chunks, self.data_chunk_length, self.num_agents, -1).to(self.device), 
			# 	rnn_hidden_state_batch.reshape(num_episodes, data_chunks, self.data_chunk_length, self.rnn_num_layers, self.num_agents, -1)[:, :, 0].permute(2, 0, 1, 3, 4).reshape(self.rnn_num_layers, num_episodes*data_chunks*self.num_agents, -1).to(self.device),
			# 	mask_actions_batch.reshape(num_episodes*data_chunks, self.data_chunk_length, self.num_agents, -1).to(self.device))
			
			# a_argmax = torch.argmax(Q_evals, dim=-1, keepdim=True)
			# Q_target = torch.gather(Q_target, dim=-1, index=a_argmax.to(self.device)).squeeze(-1)
			# Q_mix_target = target_QMix_network(
			# Q_target, 
			# full_state_batch.reshape(num_episodes*data_chunks, self.data_chunk_length, -1).to(self.device), 
			# ).reshape(-1) #* team_mask_batch.reshape(-1).to(self.device)

		
		# Finally using TD-lambda equation to generate targets
		target_Q_mix_values = self.build_td_lambda_targets(reward_batch.reshape(-1, self.max_episode_len), done_batch.reshape(-1, self.max_episode_len), Q_mix_target.reshape(-1, self.max_episode_len).cpu())
		# target_Q_mix_values = self.build_td_lambda_targets(reward_batch.reshape(-1, self.max_episode_len), done_batch.reshape(-1, self.max_episode_len), Q_mix_target.reshape(-1, self.max_episode_len).cpu(), next_Q_mix_target.reshape(-1, self.max_episode_len).cpu())

		
		state_batch = state_batch.reshape(num_episodes, data_chunks, self.data_chunk_length, self.num_agents, -1)[:, rand_time].reshape(num_episodes*data_chunks, self.data_chunk_length, self.num_agents, -1)
		rnn_hidden_state_batch = rnn_hidden_state_batch.reshape(num_episodes, data_chunks, self.data_chunk_length, self.rnn_num_layers, self.num_agents, -1)[:, rand_time][:, :, 0].permute(2, 0, 1, 3, 4).reshape(self.rnn_num_layers, num_episodes*data_chunks*self.num_agents, -1)
		full_state_batch = full_state_batch.reshape(num_episodes, data_chunks, self.data_chunk_length, 1, -1)[:, rand_time].reshape(num_episodes*data_chunks, self.data_chunk_length, -1)
		actions_batch = torch.from_numpy(np.take(self.buffer['actions'], batch_indices, axis=0)).long().reshape(num_episodes, data_chunks, self.data_chunk_length, self.num_agents, -1)[:, rand_time].reshape(num_episodes*data_chunks, self.data_chunk_length, self.num_agents, -1)
		last_one_hot_actions_batch = last_one_hot_actions_batch.reshape(num_episodes, data_chunks, self.data_chunk_length, self.num_agents, -1)[:, rand_time].reshape(num_episodes*data_chunks, self.data_chunk_length, self.num_agents, -1)
		mask_actions_batch = mask_actions_batch.reshape(num_episodes, data_chunks, self.data_chunk_length, self.num_agents, -1)[:, rand_time].reshape(num_episodes*data_chunks, self.data_chunk_length, self.num_agents, -1)
		next_state_batch = next_state_batch.reshape(num_episodes, data_chunks, self.data_chunk_length, self.num_agents, -1)[:, rand_time].reshape(num_episodes*data_chunks, self.data_chunk_length, self.num_agents, -1)
		next_rnn_hidden_state_batch = next_rnn_hidden_state_batch.reshape(num_episodes, data_chunks, self.data_chunk_length, self.rnn_num_layers, self.num_agents, -1)[:, rand_time][:, :, 0].permute(2, 0, 1, 3, 4).reshape(self.rnn_num_layers, num_episodes*data_chunks*self.num_agents, -1)
		next_full_state_batch = next_full_state_batch.reshape(num_episodes, data_chunks, self.data_chunk_length, 1, -1)[:, rand_time].reshape(num_episodes*data_chunks, self.data_chunk_length, -1)
		next_last_one_hot_actions_batch = next_last_one_hot_actions_batch.reshape(num_episodes, data_chunks, self.data_chunk_length, self.num_agents, -1)[:, rand_time].reshape(num_episodes*data_chunks, self.data_chunk_length, self.num_agents, -1)
		next_mask_actions_batch = next_mask_actions_batch.reshape(num_episodes, data_chunks, self.data_chunk_length, self.num_agents, -1)[:, rand_time].reshape(num_episodes*data_chunks, self.data_chunk_length, self.num_agents, -1)
		reward_batch = reward_batch.reshape(num_episodes, data_chunks, self.data_chunk_length, -1)[:, rand_time].reshape(num_episodes*data_chunks, self.data_chunk_length, -1)
		done_batch = done_batch.reshape(num_episodes, data_chunks, self.data_chunk_length, -1)[:, rand_time].reshape(num_episodes*data_chunks, self.data_chunk_length, -1)
		indiv_dones_batch = torch.from_numpy(np.take(self.buffer['indiv_dones'], batch_indices, axis=0)).reshape(num_episodes, data_chunks, self.data_chunk_length, self.num_agents, -1)[:, rand_time].reshape(num_episodes*data_chunks, self.data_chunk_length, self.num_agents, -1)
		next_indiv_dones_batch = torch.from_numpy(np.take(self.buffer['next_indiv_dones'], batch_indices, axis=0)).reshape(num_episodes, data_chunks, self.data_chunk_length, self.num_agents, -1)[:, rand_time].reshape(num_episodes*data_chunks, self.data_chunk_length, self.num_agents, -1)
		team_mask_batch = torch.from_numpy(np.take(self.buffer['mask'], batch_indices, axis=0)).reshape(num_episodes, data_chunks, self.data_chunk_length, -1)[:, rand_time].reshape(num_episodes*data_chunks, self.data_chunk_length, -1)

		target_Q_mix_values = target_Q_mix_values.reshape(num_episodes, data_chunks, self.data_chunk_length)[:, rand_time].reshape(num_episodes*data_chunks, self.data_chunk_length)

		max_episode_len = int(np.max(self.episode_len[batch_indices]))

		return state_batch, rnn_hidden_state_batch, full_state_batch, actions_batch, last_one_hot_actions_batch, mask_actions_batch, next_state_batch, next_rnn_hidden_state_batch, next_full_state_batch, \
		next_last_one_hot_actions_batch, next_mask_actions_batch, reward_batch, done_batch, indiv_dones_batch, next_indiv_dones_batch, team_mask_batch, max_episode_len, target_Q_mix_values

	def __len__(self):
		return self.length
