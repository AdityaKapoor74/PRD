import numpy as np 
import random
import torch
import torch.nn as nn
from functools import reduce
from torch.optim import Adam, RMSprop
import torch.nn.functional as F
from model import QMIXNetwork, RNNQNetwork
from utils import soft_update, hard_update

EPS = 1e-2

class QMIXAgent:

	def __init__(
		self, 
		env, 
		dictionary,
		comet_ml,
		):		

		# Environment Setup
		self.env = env
		self.env_name = dictionary["env"]
		self.num_agents = self.env.num_agents
		self.num_actions = self.env.action_space("pursuer_0").n

		# Training setup
		self.scheduler_need = dictionary["scheduler_need"]
		if dictionary["device"] == "gpu":
			self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		else:
			self.device = "cpu"
		self.soft_update = dictionary["soft_update"]
		self.target_update_interval = dictionary["target_update_interval"]
		self.batch_size = dictionary["batch_size"]
		self.gamma = dictionary["gamma"]
		self.num_updates = dictionary["num_updates"]
	
		# Model Setup
		self.learning_rate = dictionary["learning_rate"]
		self.grad_clip = dictionary["grad_clip"]
		self.tau = dictionary["tau"] # target network smoothing coefficient
		self.rnn_hidden_dim = dictionary["rnn_hidden_dim"]
		self.hidden_dim = dictionary["hidden_dim"]

		self.lambda_ = dictionary["lambda"]
		self.norm_returns = dictionary["norm_returns"]

		self.height = dictionary["height"]
		self.width = dictionary["width"]
		self.channel = dictionary["channel"]


		# Q Network
		self.Q_network = RNNQNetwork(self.height, self.width, self.channel, self.num_agents, self.num_actions, self.rnn_hidden_dim, self.device).to(self.device)
		self.target_Q_network = RNNQNetwork(self.height, self.width, self.channel, self.num_agents, self.num_actions, self.rnn_hidden_dim, self.device).to(self.device)
		
		self.QMix_network = QMIXNetwork(self.height, self.width, self.channel, self.num_agents, self.hidden_dim).to(self.device)
		self.target_QMix_network = QMIXNetwork(self.height, self.width, self.channel, self.num_agents, self.hidden_dim).to(self.device)

		self.loss_fn = nn.HuberLoss(reduction="sum")

		self.model_parameters = list(self.Q_network.parameters()) + list(self.QMix_network.parameters())
		self.optimizer = Adam(self.model_parameters, lr=self.learning_rate)
		
		# Loading models
		if dictionary["load_models"]:
			# For CPU
			if torch.cuda.is_available() is False:
				self.Q_network.load_state_dict(torch.load(dictionary["model_path_q_net"], map_location=torch.device('cpu')))
				self.QMix_network.load_state_dict(torch.load(dictionary["model_path_qmix_net"], map_location=torch.device('cpu')))
			# For GPU
			else:
				self.Q_network.load_state_dict(torch.load(dictionary["model_path_q_net"]))
				self.QMix_network.load_state_dict(torch.load(dictionary["model_path_qmix_net"]))

		# Copy network params
		hard_update(self.target_Q_network, self.Q_network)
		hard_update(self.target_QMix_network, self.QMix_network)
		# Disable updates for old network
		for param in self.target_Q_network.parameters():
			param.requires_grad_(False)
		for param in self.target_QMix_network.parameters():
			param.requires_grad_(False)
				

		if self.scheduler_need:
			self.scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[1000, 20000], gamma=0.1)

		self.comet_ml = None
		if dictionary["save_comet_ml_plot"]:
			self.comet_ml = comet_ml

	def get_action(self, state, last_one_hot_action, epsilon_greedy):
		if np.random.uniform() < epsilon_greedy:
			actions = [np.random.choice(self.num_actions) for _ in range(self.num_agents)]
		else:
			with torch.no_grad():
				state = torch.FloatTensor(state).permute(0, 3, 1, 2).to(self.device)
				last_one_hot_action = torch.FloatTensor(last_one_hot_action).to(self.device)
				Q_values = self.Q_network(state, last_one_hot_action)
				actions = Q_values.argmax(dim=-1).cpu().tolist()
				# actions = [Categorical(dist).sample().detach().cpu().item() for dist in Q_values]
		
		return actions


	def plot(self, episode):
		self.comet_ml.log_metric('Loss',self.plotting_dict["loss"],episode)
		self.comet_ml.log_metric('Grad_Norm',self.plotting_dict["grad_norm"],episode)

	# def calculate_deltas(self, values, rewards, dones):
	# 	deltas = []
	# 	next_value = 0
	# 	# rewards = rewards.unsqueeze(-1)
	# 	# dones = dones.unsqueeze(-1)
	# 	masks = 1-dones
	# 	for t in reversed(range(0, len(rewards))):
	# 		td_error = rewards[t] + (self.gamma * next_value * masks[t]) - values.data[t]
	# 		next_value = values.data[t]
	# 		deltas.insert(0,td_error)
	# 	deltas = torch.stack(deltas)

	# 	return deltas


	# def nstep_returns(self, values, rewards, dones):
	# 	deltas = self.calculate_deltas(values, rewards, dones)
	# 	advs = self.calculate_returns(deltas, self.gamma*self.lambda_)
	# 	target_Vs = advs+values
	# 	return target_Vs


	def calculate_returns(self, rewards):
		returns = []
		R = 0
		
		for r in reversed(rewards):
			R = r + R * self.gamma
			returns.insert(0, R)
		
		returns_tensor = torch.stack(returns)
		
		if self.norm_returns:
			returns_tensor = (returns_tensor - returns_tensor.mean()) / returns_tensor.std()
			
		return returns_tensor

	# def TD_error(self, target_values, values, rewards, dones, mask):
	# 	TD_errors = []
	# 	curr_td_error = 0
	# 	for t in reversed(range(0, len(rewards))):
	# 		td_error = ((rewards[t] + (self.gamma * target_values[t] * (1 - dones[t])))*mask[t] - values[t])**2
	# 		curr_td_error = td_error + (self.gamma * self.lambda_ * curr_td_error * (1 - dones[t]))
	# 		TD_errors.insert(0, curr_td_error)

	# 	TD_errors = torch.sum(torch.stack(TD_errors)) / mask.sum()
		
	# 	return TD_errors

	def build_td_lambda_targets(self, rewards, terminated, mask, target_qs):
		# Assumes  <target_qs > in B*T*A and <reward >, <terminated >  in B*T*A, <mask > in (at least) B*T-1*1
		# Initialise  last  lambda -return  for  not  terminated  episodes
		ret = target_qs.new_zeros(*target_qs.shape)
		ret = target_qs * (1-terminated)
		# ret[:, -1] = target_qs[:, -1] * (1 - (torch.sum(terminated, dim=1)>0).int())
		# Backwards  recursive  update  of the "forward  view"
		for t in range(ret.shape[1] - 2, -1,  -1):
			ret[:, t] = self.lambda_ * self.gamma * ret[:, t + 1] + mask[:, t] \
						* (rewards[:, t] + (1 - self.lambda_) * self.gamma * target_qs[:, t + 1] * (1 - terminated[:, t]))
		# Returns lambda-return from t=0 to t=T-1, i.e. in B*T-1*A
		# return ret[:, 0:-1]
		return ret

	def update(self, sample, episode):
		# # sample episodes from replay buffer
		state_batch, actions_batch, last_one_hot_actions_batch, next_state_batch, next_last_one_hot_actions_batch, reward_batch, done_batch, mask_batch, max_episode_len = sample
		# # convert list to tensor
		state_batch = torch.FloatTensor(state_batch)
		actions_batch = torch.FloatTensor(actions_batch).long()
		last_one_hot_actions_batch = torch.FloatTensor(last_one_hot_actions_batch)
		next_state_batch = torch.FloatTensor(next_state_batch)
		next_last_one_hot_actions_batch = torch.FloatTensor(next_last_one_hot_actions_batch)
		reward_batch = torch.FloatTensor(reward_batch)
		done_batch = torch.FloatTensor(done_batch).long()
		mask_batch = torch.FloatTensor(mask_batch).long()

		# final_state = torch.cat([state_batch, last_one_hot_actions_batch], dim=-1)

		# Q_values = self.Q_network(final_state.to(self.device))
		# Q_a_values = torch.gather(Q_values, dim=-1, index=actions_batch.unsqueeze(-1).to(self.device)).squeeze(-1)
		# Q_mix_values = self.QMix_network(Q_a_values, state_batch.reshape(state_batch.shape[0], state_batch.shape[1], -1).to(self.device)).reshape(state_batch.shape[0], state_batch.shape[1], 1)*mask_batch.unsqueeze(-1).to(self.device)

		# # Calcuate Q targets with TD-lambda
		# with torch.no_grad():
		# 	target_Q_values = self.target_Q_network(final_state.to(self.device))
		# 	target_Q_a_values = torch.gather(target_Q_values, dim=-1, index=actions_batch.unsqueeze(-1).to(self.device)).squeeze(-1)
		# 	target_Q_mix_values = self.target_QMix_network(target_Q_a_values, state_batch.reshape(state_batch.shape[0], state_batch.shape[1], -1).to(self.device)).reshape(state_batch.shape[0], state_batch.shape[1], 1)
		# 	target_Q_mix_values = self.build_td_lambda_targets(reward_batch.unsqueeze(-1).to(self.device), done_batch.unsqueeze(-1).to(self.device), mask_batch.to(self.device), target_Q_mix_values)

		# Q_loss = self.loss_fn(Q_mix_values, target_Q_mix_values) / torch.sum(mask_batch)

		# self.optimizer.zero_grad()
		# Q_loss.backward()
		# grad_norm = torch.nn.utils.clip_grad_norm_(self.model_parameters, self.grad_clip).item()
		# # grad_norm = -1
		# self.optimizer.step()

		Q_loss_batch = 0.0

		self.Q_network.rnn_hidden_state = None
		self.target_Q_network.rnn_hidden_state = None

		Q_mix_values = []
		target_Q_mix_values = []

		for _ in range(self.num_updates):
			for t in range(max_episode_len):
				# train in time order
				states_slice = state_batch[:,t].reshape(-1, self.height, self.width, self.channel).permute(0, 3, 1, 2)
				last_one_hot_actions_slice = last_one_hot_actions_batch[:,t].reshape(-1, self.num_actions)
				actions_slice = actions_batch[:, t].reshape(-1)
				next_states_slice = next_state_batch[:,t].reshape(-1, self.height, self.width, self.channel).permute(0, 3, 1, 2)
				next_last_one_hot_actions_slice = next_last_one_hot_actions_batch[:,t].reshape(-1, self.num_actions)
				reward_slice = reward_batch[:, t].reshape(-1)
				done_slice = done_batch[:, t].reshape(-1)
				mask_slice = mask_batch[:, t].reshape(-1)

				if mask_slice.sum().cpu().numpy() < EPS:
					break

				Q_values = self.Q_network(states_slice.to(self.device), last_one_hot_actions_slice.to(self.device))
				Q_evals = torch.gather(Q_values, dim=-1, index=actions_slice.unsqueeze(-1).to(self.device)).squeeze(-1)
				Q_mix = self.QMix_network(Q_evals, state_batch[:,t].reshape(-1, self.height, self.width, self.channel).permute(0, 3, 1, 2).to(self.device)).squeeze(-1).squeeze(-1) * mask_slice.to(self.device)

				with torch.no_grad():
					Q_evals_next = self.Q_network(next_states_slice.to(self.device), next_last_one_hot_actions_slice.to(self.device))
					Q_targets = self.target_Q_network(next_states_slice.to(self.device), next_last_one_hot_actions_slice.to(self.device))
					a_argmax = torch.argmax(Q_evals_next, dim=-1, keepdim=True)
					Q_targets = torch.gather(Q_targets, dim=-1, index=a_argmax.to(self.device)).squeeze(-1)
					Q_mix_target = self.target_QMix_network(Q_targets, next_state_batch[:, t].reshape(-1, self.height, self.width, self.channel).permute(0, 3, 1, 2).to(self.device)).squeeze(-1).squeeze(-1)
					
				Q_mix_values.append(Q_mix)
				target_Q_mix_values.append(Q_mix_target)

				# Q_loss = self.TD_error(Q_mix_target, Q_mix_values, reward_slice.to(self.device), done_slice.to(self.device), mask_slice.to(self.device))

				# Q_loss_batch += Q_loss.item()

				# self.optimizer.zero_grad()
				# Q_loss.backward()
				# grad_norm = torch.nn.utils.clip_grad_norm_(self.model_parameters, self.grad_clip).item()
				# # grad_norm = -1
				# self.optimizer.step()

		# Q_loss_batch /= (max_episode_len*self.num_updates)

		Q_mix_values = torch.stack(Q_mix_values, dim=1).to(self.device)
		target_Q_mix_values = torch.stack(target_Q_mix_values, dim=1).to(self.device)

		target_Q_mix_values = self.build_td_lambda_targets(reward_batch.to(self.device), done_batch.to(self.device), mask_batch.to(self.device), target_Q_mix_values)

		Q_loss = self.loss_fn(Q_mix_values, target_Q_mix_values.detach()) / mask_batch.to(self.device).sum()

		self.optimizer.zero_grad()
		Q_loss.backward()
		grad_norm = torch.nn.utils.clip_grad_norm_(self.model_parameters, self.grad_clip).item()
		# grad_norm = 0
		# for p in self.model_parameters:
		# 	param_norm = p.grad.detach().data.norm(2)
		# 	grad_norm += param_norm.item() ** 2
		# grad_norm = torch.tensor(grad_norm) ** 0.5
		self.optimizer.step()

		if self.scheduler_need:
			self.scheduler.step()

		if self.soft_update:
			soft_update(self.target_Q_network, self.Q_network, self.tau)
			soft_update(self.target_QMix_network, self.QMix_network, self.tau)
		else:
			if episode % self.target_update_interval == 0:
				hard_update(self.target_Q_network, self.Q_network)
				hard_update(self.target_QMix_network, self.QMix_network)


		self.plotting_dict = {
		"loss": Q_loss.item(),
		"grad_norm": grad_norm,
		}

		if self.comet_ml is not None:
			self.plot(episode)

		torch.cuda.empty_cache()      
