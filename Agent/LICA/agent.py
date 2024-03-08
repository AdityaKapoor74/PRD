import numpy as np 
import random
import torch
import torch.nn as nn
from functools import reduce
from torch.optim import AdamW
import torch.nn.functional as F
from model import LICACritic, RNNAgent
from utils import soft_update, hard_update, GumbelSoftmax, multinomial_entropy

EPS = 1e-2

class LICAAgent:

	def __init__(
		self, 
		env, 
		dictionary,
		comet_ml,
		):		

		# Environment Setup
		self.env = env
		self.env_name = dictionary["env"]
		self.num_agents = self.env.n_agents
		self.num_actions = self.env.action_space[0].n
		self.critic_obs_shape = dictionary["global_observation"]
		self.actor_obs_shape = dictionary["local_observation"]

		# Training setup
		self.scheduler_need = dictionary["scheduler_need"]
		if dictionary["device"] == "gpu":
			self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		else:
			self.device = "cpu"
		self.soft_update = dictionary["soft_update"]
		self.target_update_interval = dictionary["target_update_interval"]
		self.entropy_coeff = dictionary["entropy_coeff"]
		self.gamma = dictionary["gamma"]
		self.num_updates = dictionary["num_updates"]
		self.max_time_steps = dictionary["max_time_steps"]
		self.update_episode_interval = dictionary["update_episode_interval"]
	
		# Model Setup
		self.enable_grad_clip_actor = dictionary["enable_grad_clip_actor"]
		self.enable_grad_clip_critic = dictionary["enable_grad_clip_critic"]
		self.critic_learning_rate = dictionary["critic_learning_rate"]
		self.actor_learning_rate = dictionary["actor_learning_rate"]
		self.critic_grad_clip = dictionary["critic_grad_clip"]
		self.actor_grad_clip = dictionary["actor_grad_clip"]
		self.tau = dictionary["tau"] # target network smoothing coefficient
		self.rnn_num_layers = dictionary["rnn_num_layers"]
		self.rnn_hidden_dim = dictionary["rnn_hidden_dim"]
		self.mixing_embed_dim = dictionary["mixing_embed_dim"]
		self.num_hypernet_layers = dictionary["num_hypernet_layers"]

		self.lambda_ = dictionary["lambda"]
		self.norm_returns = dictionary["norm_returns"]

		# Q Network
		self.actor = RNNAgent(self.actor_obs_shape, self.rnn_num_layers, self.rnn_hidden_dim, self.num_actions).to(self.device)
		
		self.critic = LICACritic(self.critic_obs_shape, self.mixing_embed_dim, self.num_actions, self.num_agents, self.num_hypernet_layers).to(self.device)
		self.target_critic = LICACritic(self.critic_obs_shape, self.mixing_embed_dim, self.num_actions, self.num_agents, self.num_hypernet_layers).to(self.device)

		self.loss_fn = nn.HuberLoss(reduction="sum")

		self.critic_optimizer = AdamW(self.critic.parameters(), lr=self.critic_learning_rate, weight_decay=dictionary["critic_weight_decay"], eps=1e-05)
		self.actor_optimizer = AdamW(self.actor.parameters(), lr=self.actor_learning_rate, weight_decay=dictionary["actor_weight_decay"], eps=1e-05)
		
		# Loading models
		if dictionary["load_models"]:
			# For CPU
			if torch.cuda.is_available() is False:
				self.actor.load_state_dict(torch.load(dictionary["model_path_actor"], map_location=torch.device('cpu')))
				self.critic.load_state_dict(torch.load(dictionary["model_path_critic"], map_location=torch.device('cpu')))
			# For GPU
			else:
				self.actor.load_state_dict(torch.load(dictionary["model_path_actor"]))
				self.critic.load_state_dict(torch.load(dictionary["model_path_critic"]))

		# Copy network params
		hard_update(self.target_critic, self.critic)
		# Disable updates for old network
		for param in self.target_critic.parameters():
			param.requires_grad_(False)
				

		if self.scheduler_need:
			self.critic_scheduler = optim.lr_scheduler.MultiStepLR(self.critic_optimizer, milestones=[1000, 20000], gamma=0.1)
			self.actor_scheduler = optim.lr_scheduler.MultiStepLR(self.actor_optimizer, milestones=[1000, 20000], gamma=0.1)

		self.comet_ml = None
		if dictionary["save_comet_ml_plot"]:
			self.comet_ml = comet_ml

	def get_action(self, state, rnn_hidden_state, mask_actions):
		with torch.no_grad():
			state = torch.from_numpy(state).float().unsqueeze(0).unsqueeze(0)
			rnn_hidden_state = torch.from_numpy(rnn_hidden_state).float()
			mask_actions = torch.from_numpy(mask_actions).bool()
			dists, h = self.actor(state.to(self.device), rnn_hidden_state.to(self.device), mask_actions.to(self.device))
			actions = GumbelSoftmax(logits=dists.reshape(self.num_agents, self.num_actions)).sample()
			actions = torch.argmax(actions, dim=-1).tolist()
		
		return actions, h.cpu().numpy()


	def plot(self, episode):
		self.comet_ml.log_metric('Actor Loss',self.plotting_dict["actor_loss"],episode)
		self.comet_ml.log_metric('Actor Grad Norm',self.plotting_dict["actor_grad_norm"],episode)

		self.comet_ml.log_metric('Critic Loss',self.plotting_dict["critic_loss"],episode)
		self.comet_ml.log_metric('Critic Grad Norm',self.plotting_dict["critic_grad_norm"],episode)

		self.comet_ml.log_metric('Entropy',self.plotting_dict["entropy"],episode)


	# def build_td_lambda_targets(self, rewards, terminated, mask, target_qs):
	# 	# Assumes  <target_qs > in B*T*A and <reward >, <terminated >  in B*T*A, <mask > in (at least) B*T-1*1
	# 	# Initialise  last  lambda -return  for  not  terminated  episodes
	# 	ret = target_qs.new_zeros(*target_qs.shape)
	# 	ret = target_qs * (1-terminated[:, 1:])
	# 	# ret[:, -1] = target_qs[:, -1] * (1 - (torch.sum(terminated, dim=1)>0).int())
	# 	# Backwards  recursive  update  of the "forward  view"
	# 	for t in range(ret.shape[1] - 2, -1,  -1):
	# 		ret[:, t] = self.lambda_ * self.gamma * ret[:, t + 1] + mask[:, t] \
	# 					* (rewards[:, t] + (1 - self.lambda_) * self.gamma * target_qs[:, t + 1] * (1 - terminated[:, t+1]))
	# 	# Returns lambda-return from t=0 to t=T-1, i.e. in B*T-1*A
	# 	# return ret[:, 0:-1]
	# 	return ret

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
	# 	ret[:, -1] = (rewards[:, T-1] + next_q_values[:, T-1]) * (1 - terminations[:, -1])  # Terminal values for the last timestep

	# 	# Backward recursive update of the TD-lambda targets
	# 	for t in reversed(range(T-1)):
	# 		td_error = rewards[:, t] + self.gamma * next_q_values[:, t] * (1 - terminations[:, t + 1]) - q_values[:, t]
	# 		ret[:, t] = q_values[:, t] + td_error * (1 - terminations[:, t + 1]) + self.lambda_ * self.gamma * ret[:, t + 1] * (1 - terminations[:, t + 1])

	# 	return ret

	def update(self, buffer, episode):
		
		# convert list to tensor
		# critic_state_batch = torch.FloatTensor(np.array(buffer.critic_states))
		# actor_state_batch = torch.FloatTensor(np.array(buffer.actor_states))
		# one_hot_actions_batch = torch.FloatTensor(np.array(buffer.one_hot_actions))
		# actions_batch = torch.FloatTensor(np.array(buffer.actions)).long()
		# last_one_hot_actions_batch = torch.FloatTensor(np.array(buffer.last_one_hot_actions))
		# mask_actions = torch.FloatTensor(np.array(buffer.mask_actions))
		# reward_batch = torch.FloatTensor(np.array(buffer.rewards))
		# done_batch = torch.FloatTensor(np.array(buffer.dones)).long()
		# mask_batch = torch.FloatTensor(np.array(buffer.masks)).long()


		buffer.calculate_targets(self.target_critic)

		for _ in range(self.num_updates):

			state_batch, rnn_hidden_state_batch, full_state_batch, actions_batch, one_hot_actions_batch, action_masks_batch, next_full_state_batch, \
			next_one_hot_actions_batch, reward_batch, done_batch, next_done_batch, mask_batch, TD_target_Qs_batch = buffer.sample()


			Qs = self.critic(one_hot_actions_batch.reshape(self.update_episode_interval, self.max_time_steps, self.num_agents, self.num_actions).to(self.device), full_state_batch.reshape(self.update_episode_interval, self.max_time_steps, -1).to(self.device)).squeeze(-1)
			Qs *= (1-done_batch).reshape(self.update_episode_interval, self.max_time_steps).to(self.device)
			
			# target_Qs = self.target_critic(one_hot_actions_batch.reshape(self.update_episode_interval, self.max_time_steps, self.num_agents, self.num_actions).to(self.device), full_state_batch.reshape(self.update_episode_interval, self.max_time_steps, -1).to(self.device)).squeeze(-1)
			# next_target_Qs = self.target_critic(next_one_hot_actions_batch.reshape(self.update_episode_interval, self.max_time_steps, self.num_agents, self.num_actions).to(self.device), next_full_state_batch.reshape(self.update_episode_interval, self.max_time_steps, -1).to(self.device)).squeeze(-1)
			# # TD_target_Qs = self.build_td_lambda_targets(reward_batch.to(self.device), done_batch.to(self.device), mask_batch.to(self.device), target_Qs)
			# TD_target_Qs = self.build_td_lambda_targets(reward_batch.reshape(self.update_episode_interval, self.max_time_steps).to(self.device), done_batch.reshape(self.update_episode_interval, self.max_time_steps).to(self.device), target_Qs, next_target_Qs)
			# TD_target_Qs *= (1-done_batch).reshape(self.update_episode_interval, self.max_time_steps).to(self.device)

			Q_loss = self.loss_fn(Qs, TD_target_Qs_batch.to(self.device)) / mask_batch.to(self.device).sum()

			self.critic_optimizer.zero_grad()
			Q_loss.backward()
			if self.enable_grad_clip_critic:
				critic_grad_norm = torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.critic_grad_clip).item()
			else:
				grad_norm = 0
				for p in self.critic.parameters():
					param_norm = p.grad.detach().data.norm(2)
					grad_norm += param_norm.item() ** 2
				critic_grad_norm = torch.tensor(grad_norm) ** 0.5
			self.critic_optimizer.step()

			dists, _ = self.actor(state_batch.to(self.device), rnn_hidden_state_batch.to(self.device), action_masks_batch.to(self.device))
			probs = F.softmax(dists, dim=-1) * (1-done_batch).unsqueeze(-1).to(self.device)
			entropy = multinomial_entropy(dists).mean(dim=-1, keepdim=True)
			

			# self.actor.rnn_hidden_obs = None
			# probs = []
			# entropy = []

			# for t in range(mask_batch.shape[1]):
			# 	# train in time order
			# 	mask_slice = mask_batch[:, t].reshape(-1)

			# 	# if mask_slice.sum().cpu().numpy() < EPS:
			# 	# 	break

			# 	actor_states_slice = actor_state_batch[:,t].reshape(-1, self.actor_obs_shape)
			# 	last_one_hot_action_slice = last_one_hot_actions_batch[:, t].reshape(-1, self.num_actions)
			# 	mask_actions_slice = mask_actions[:, t].reshape(-1, self.num_actions)
				
			# 	final_state = torch.cat([actor_states_slice, last_one_hot_action_slice], dim=-1)
			# 	dist = self.actor(final_state.to(self.device))
			# 	ent = multinomial_entropy(dist+mask_actions_slice.to(self.device)).mean(dim=-1, keepdim=True)
			# 	prob = F.softmax(dist + mask_actions_slice.to(self.device), dim=-1)

			# 	probs.append(prob)
			# 	entropy.append(ent)

			# probs = torch.stack(probs, dim=1).reshape(-1, mask_batch.shape[1], self.num_agents, self.num_actions)
			# entropy = torch.stack(entropy, dim=1).reshape(-1, mask_batch.shape[1])

			mix_loss = self.critic(probs.reshape(self.update_episode_interval, self.max_time_steps, self.num_agents, self.num_actions).to(self.device), full_state_batch.reshape(self.update_episode_interval, self.max_time_steps, -1).to(self.device)).squeeze(-1)

			mix_loss = (mix_loss * (1-done_batch).reshape(self.update_episode_interval, self.max_time_steps).to(self.device)).sum() / (1-done_batch).sum().to(self.device)

			# adaptive entropy
			entropy_loss = (entropy.to(self.device) * (1-done_batch).to(self.device)).sum() / (1-done_batch).sum().to(self.device)
			entropy_coeff = self.entropy_coeff / entropy_loss.item()
			entropy_coeff = self.entropy_coeff

			mix_loss = - mix_loss - entropy_coeff*entropy_loss

			self.actor_optimizer.zero_grad()
			mix_loss.backward()
			if self.enable_grad_clip_actor:
				actor_grad_norm = torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.actor_grad_clip).item()
			else:
				grad_norm = 0
				for p in self.actor.parameters():
					param_norm = p.grad.detach().data.norm(2)
					grad_norm += param_norm.item() ** 2
				actor_grad_norm = torch.tensor(grad_norm) ** 0.5
			self.actor_optimizer.step()
			

		if self.scheduler_need:
			self.actor_scheduler.step()
			self.critic_scheduler.step()

		if self.soft_update:
			soft_update(self.target_critic, self.critic, self.tau)
		else:
			if episode % self.target_update_interval == 0:
				hard_update(self.target_critic, self.critic)


		self.plotting_dict = {
		"actor_loss": mix_loss.item(),
		"critic_loss": Q_loss.item(),
		"actor_grad_norm": actor_grad_norm,
		"critic_grad_norm": critic_grad_norm,
		"entropy": entropy_loss.item(),
		}

		if self.comet_ml is not None:
			self.plot(episode)

		torch.cuda.empty_cache()      
