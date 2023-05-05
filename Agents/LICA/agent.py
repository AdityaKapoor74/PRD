import numpy as np 
import random
import torch
import torch.nn as nn
from functools import reduce
from torch.optim import Adam
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
		self.obs_input_dim = dictionary["observation_shape"] # crossing_team_greedy

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
	
		# Model Setup
		self.critic_learning_rate = dictionary["critic_learning_rate"]
		self.actor_learning_rate = dictionary["actor_learning_rate"]
		self.critic_grad_clip = dictionary["critic_grad_clip"]
		self.actor_grad_clip = dictionary["actor_grad_clip"]
		self.tau = dictionary["tau"] # target network smoothing coefficient
		self.rnn_hidden_dim = dictionary["rnn_hidden_dim"]
		self.mixing_embed_dim = dictionary["mixing_embed_dim"]
		self.num_hypernet_layers = dictionary["num_hypernet_layers"]

		self.lambda_ = dictionary["lambda"]
		self.norm_returns = dictionary["norm_returns"]

		# Q Network
		self.actor = RNNAgent(self.obs_input_dim, self.rnn_hidden_dim, self.num_actions).to(self.device)
		
		self.critic = LICACritic(self.obs_input_dim, self.mixing_embed_dim, self.num_actions, self.num_agents, self.num_hypernet_layers).to(self.device)
		self.target_critic = LICACritic(self.obs_input_dim, self.mixing_embed_dim, self.num_actions, self.num_agents, self.num_hypernet_layers).to(self.device)

		self.loss_fn = nn.HuberLoss(reduction="sum")

		self.critic_optimizer = Adam(self.critic.parameters(), lr=self.critic_learning_rate)
		self.actor_optimizer = Adam(self.actor.parameters(), lr=self.actor_learning_rate)
		
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

	def get_action(self, state, last_one_hot_action):
		with torch.no_grad():
			state = torch.FloatTensor(state)
			last_one_hot_action = torch.FloatTensor(last_one_hot_action)
			final_state = torch.cat([state, last_one_hot_action], dim=-1).to(self.device)
			dists = self.actor(final_state)
			actions = GumbelSoftmax(logits=dists).sample()
			actions = torch.argmax(actions, dim=-1).tolist()
		
		return actions


	def plot(self, episode):
		self.comet_ml.log_metric('Actor Loss',self.plotting_dict["actor_loss"],episode)
		self.comet_ml.log_metric('Actor Grad Norm',self.plotting_dict["actor_grad_norm"],episode)

		self.comet_ml.log_metric('Critic Loss',self.plotting_dict["critic_loss"],episode)
		self.comet_ml.log_metric('Critic Grad Norm',self.plotting_dict["critic_grad_norm"],episode)

		self.comet_ml.log_metric('Entropy',self.plotting_dict["entropy"],episode)


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

	def update(self, buffer, episode):
		
		# # convert list to tensor
		state_batch = torch.FloatTensor(np.array(buffer.states))
		one_hot_actions_batch = torch.FloatTensor(np.array(buffer.one_hot_actions))
		actions_batch = torch.FloatTensor(np.array(buffer.actions)).long()
		last_one_hot_actions_batch = torch.FloatTensor(np.array(buffer.last_one_hot_actions))
		reward_batch = torch.FloatTensor(np.array(buffer.rewards))
		done_batch = torch.FloatTensor(np.array(buffer.dones)).long()
		mask_batch = torch.FloatTensor(np.array(buffer.masks)).long()


		for _ in range(self.num_updates):

			Qs = self.critic(one_hot_actions_batch.to(self.device), state_batch.to(self.device)).squeeze(-1) * mask_batch.to(self.device)
			
			target_Qs = self.target_critic(one_hot_actions_batch.to(self.device), state_batch.to(self.device)).squeeze(-1)
			target_Qs = self.build_td_lambda_targets(reward_batch.to(self.device), done_batch.to(self.device), mask_batch.to(self.device), target_Qs)

			Q_loss = self.loss_fn(Qs, target_Qs.detach()) / mask_batch.to(self.device).sum()

			self.critic_optimizer.zero_grad()
			Q_loss.backward()
			critic_grad_norm = torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.critic_grad_clip).item()
			# grad_norm = 0
			# for p in self.model_parameters:
			# 	param_norm = p.grad.detach().data.norm(2)
			# 	grad_norm += param_norm.item() ** 2
			# grad_norm = torch.tensor(grad_norm) ** 0.5
			self.critic_optimizer.step()

			self.actor.rnn_hidden_obs = None
			probs = []
			entropy = []

			for t in range(mask_batch.shape[1]):
				# train in time order
				mask_slice = mask_batch[:, t].reshape(-1)

				if mask_slice.sum().cpu().numpy() < EPS:
					break

				states_slice = state_batch[:,t].reshape(-1, self.obs_input_dim)
				last_one_hot_action_slice = last_one_hot_actions_batch[:, t].reshape(-1, self.num_actions)
				
				final_state = torch.cat([states_slice, last_one_hot_action_slice], dim=-1)
				dist = self.actor(final_state.to(self.device))
				ent = multinomial_entropy(dist).mean(dim=-1, keepdim=True)
				prob = F.softmax(dist, dim=-1)

				probs.append(prob)
				entropy.append(ent)

			probs = torch.stack(probs, dim=1).reshape(-1, mask_batch.shape[1], self.num_agents, self.num_actions)
			entropy = torch.stack(entropy, dim=1).reshape(-1, mask_batch.shape[1])

			mix_loss = self.critic(probs.to(self.device), state_batch.to(self.device)).squeeze(-1)

			mix_loss = (mix_loss * mask_batch.to(self.device)).sum() / mask_batch.sum().to(self.device)

			# adaptive entropy
			entropy_loss = (entropy.to(self.device) * mask_batch.to(self.device)).sum() / mask_batch.sum().to(self.device)
			entropy_coeff = self.entropy_coeff / entropy_loss.item()
			# entropy_coeff = self.entropy_coeff

			mix_loss = -mix_loss - entropy_coeff*entropy_loss

			self.actor_optimizer.zero_grad()
			mix_loss.backward()
			actor_grad_norm = torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.actor_grad_clip).item()
			# grad_norm = 0
			# for p in self.model_parameters:
			# 	param_norm = p.grad.detach().data.norm(2)
			# 	grad_norm += param_norm.item() ** 2
			# grad_norm = torch.tensor(grad_norm) ** 0.5
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
