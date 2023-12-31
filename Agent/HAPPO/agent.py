import numpy as np
import math
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from model import Policy, Q_network
from utils import RolloutBuffer

class PPOAgent:

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
		self.num_enemies = self.env.n_enemies
		self.num_actions = self.env.action_space[0].n

		# Training setup
		self.max_episodes = dictionary["max_episodes"]
		self.test_num = dictionary["test_num"]
		self.gif = dictionary["gif"]
		self.experiment_type = dictionary["experiment_type"]
		self.n_epochs = dictionary["n_epochs"]
		self.scheduler_need = dictionary["scheduler_need"]
		self.norm_rewards = dictionary["norm_rewards"]
		if dictionary["device"] == "gpu":
			self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		else:
			self.device = "cpu"

		self.update_ppo_agent = dictionary["update_ppo_agent"]
		self.max_time_steps = dictionary["max_time_steps"]

		# Critic Setup
		self.temperature_q = dictionary["temperature_q"]
		self.rnn_num_layers_q = dictionary["rnn_num_layers_q"]
		self.rnn_hidden_q = dictionary["rnn_hidden_q"]
		self.critic_ally_observation = dictionary["ally_observation"]
		self.critic_enemy_observation = dictionary["enemy_observation"]
		self.q_value_lr = dictionary["q_value_lr"]
		self.q_weight_decay = dictionary["q_weight_decay"]
		self.critic_weight_entropy_pen = dictionary["critic_weight_entropy_pen"]
		self.critic_weight_entropy_pen_final = dictionary["critic_weight_entropy_pen_final"]
		self.critic_weight_entropy_pen_decay_rate = (dictionary["critic_weight_entropy_pen_final"] - dictionary["critic_weight_entropy_pen"]) / dictionary["critic_weight_entropy_pen_steps"]
		self.critic_score_regularizer = dictionary["critic_score_regularizer"]
		self.target_calc_style = dictionary["target_calc_style"]
		self.td_lambda = dictionary["td_lambda"] # TD lambda
		self.n_steps = dictionary["n_steps"]
		self.value_clip = dictionary["value_clip"]
		self.num_heads = dictionary["num_heads"]
		self.enable_grad_clip_critic_q = dictionary["enable_grad_clip_critic_q"]
		self.grad_clip_critic_q = dictionary["grad_clip_critic_q"]
		self.norm_returns_q = dictionary["norm_returns_q"]
		self.clamp_rewards = dictionary["clamp_rewards"]
		self.clamp_rewards_value_min = dictionary["clamp_rewards_value_min"]
		self.clamp_rewards_value_max = dictionary["clamp_rewards_value_max"]


		# Actor Setup
		self.data_chunk_length = dictionary["data_chunk_length"]
		self.rnn_num_layers_actor = dictionary["rnn_num_layers_actor"]
		self.rnn_hidden_actor = dictionary["rnn_hidden_actor"]
		self.actor_observation_shape = dictionary["local_observation"]
		self.policy_lr = dictionary["policy_lr"]
		self.policy_weight_decay = dictionary["policy_weight_decay"]
		self.gamma = dictionary["gamma"]
		self.entropy_pen = dictionary["entropy_pen"]
		self.entropy_pen_decay = (dictionary["entropy_pen"] - dictionary["entropy_pen_final"])/dictionary["entropy_pen_steps"]
		self.entropy_pen_final = dictionary["entropy_pen_final"]
		self.gae_lambda = dictionary["gae_lambda"]
		self.norm_adv = dictionary["norm_adv"]
		self.policy_clip = dictionary["policy_clip"]
		self.enable_grad_clip_actor = dictionary["enable_grad_clip_actor"]
		self.grad_clip_actor = dictionary["grad_clip_actor"]
		
		# Q-V Network
		self.critic_network_q = Q_network(
			ally_obs_input_dim=self.critic_ally_observation, 
			enemy_obs_input_dim=self.critic_enemy_observation, 
			num_heads=self.num_heads, 
			num_agents=self.num_agents, 
			num_enemies=self.num_enemies, 
			num_actions=self.num_actions, 
			rnn_num_layers=self.rnn_num_layers_q,
			device=self.device, 
			attention_dropout_prob=dictionary["attention_dropout_prob_q"], 
			temperature=self.temperature_q
			).to(self.device)
		self.target_critic_network_q = Q_network(
			ally_obs_input_dim=self.critic_ally_observation, 
			enemy_obs_input_dim=self.critic_enemy_observation, 
			num_heads=self.num_heads, 
			num_agents=self.num_agents, 
			num_enemies=self.num_enemies, 
			num_actions=self.num_actions, 
			rnn_num_layers=self.rnn_num_layers_q,
			device=self.device, 
			attention_dropout_prob=dictionary["attention_dropout_prob_q"], 
			temperature=self.temperature_q
			).to(self.device)

		# Policy Network
		self.policy_network = []
		for i in range(self.num_agents):
			self.policy_network.append(Policy(
				obs_input_dim=self.actor_observation_shape, 
				num_agents=self.num_agents, 
				num_actions=self.num_actions, 
				rnn_num_layers=self.rnn_num_layers_actor,
				device=self.device
				).to(self.device))
		

		self.network_update_interval_q = dictionary["network_update_interval_q"]
		self.soft_update_q = dictionary["soft_update_q"]
		self.tau_q = dictionary["tau_q"]


		self.buffer = RolloutBuffer(
			num_episodes=self.update_ppo_agent, 
			max_time_steps=self.max_time_steps, 
			num_agents=self.num_agents, 
			num_enemies=self.num_enemies,
			obs_shape_critic_ally=self.critic_ally_observation, 
			obs_shape_critic_enemy=self.critic_enemy_observation, 
			obs_shape_actor=self.actor_observation_shape, 
			rnn_num_layers_actor=self.rnn_num_layers_actor,
			actor_hidden_state=self.rnn_hidden_actor,
			rnn_num_layers_q=self.rnn_num_layers_q,
			q_hidden_state=self.rnn_hidden_q,
			num_actions=self.num_actions,
			data_chunk_length=self.data_chunk_length,
			norm_returns_q=self.norm_returns_q,
			clamp_rewards=self.clamp_rewards,
			clamp_rewards_value_min=self.clamp_rewards_value_min,
			clamp_rewards_value_max=self.clamp_rewards_value_max,
			norm_rewards=self.norm_rewards,
			target_calc_style=self.target_calc_style,
			td_lambda=self.td_lambda,
			gae_lambda=self.gae_lambda,
			n_steps=self.n_steps,
			gamma=self.gamma,
			Q_PopArt=self.critic_network_q.q_value_layer[-1],
			)

		
		self.q_critic_optimizer = optim.AdamW(self.critic_network_q.parameters(), lr=self.q_value_lr, weight_decay=self.q_weight_decay, eps=1e-05)
		
		self.policy_optimizer = []
		for agent_id in range(self.num_agents):
			self.policy_optimizer.append(optim.AdamW(self.policy_network[agent_id].parameters(),lr=self.policy_lr, weight_decay=self.policy_weight_decay, eps=1e-05))

		if self.scheduler_need:
			self.scheduler_policy = optim.lr_scheduler.MultiStepLR(self.policy_optimizer, milestones=[1000, 20000], gamma=0.1)
			self.scheduler_q_critic = optim.lr_scheduler.MultiStepLR(self.q_critic_optimizer, milestones=[1000, 20000], gamma=0.1)

		self.comet_ml = None
		if dictionary["save_comet_ml_plot"]:
			self.comet_ml = comet_ml


		# Loading models
		if dictionary["load_models"]:
			# For CPU
			if torch.cuda.is_available() is False:
				self.critic_network_q.load_state_dict(torch.load(dictionary["model_path_value"], map_location=torch.device('cpu')))
				self.policy_network.load_state_dict(torch.load(dictionary["model_path_policy"], map_location=torch.device('cpu')))
				self.q_critic_optimizer.load_state_dict(torch.load(dictionary["model_path_q_critic_optimizer"], map_location=torch.device('cpu')))
				self.policy_optimizer.load_state_dict(torch.load(dictionary["model_path_policy_optimizer"], map_location=torch.device('cpu')))
			# For GPU
			else:
				self.critic_network_q.load_state_dict(torch.load(dictionary["model_path_value"]))
				self.policy_network.load_state_dict(torch.load(dictionary["model_path_policy"]))
				self.q_critic_optimizer.load_state_dict(torch.load(dictionary["model_path_q_critic_optimizer"]))
				self.policy_optimizer.load_state_dict(torch.load(dictionary["model_path_policy_optimizer"]))

		# Copy network params
		self.target_critic_network_q.load_state_dict(self.critic_network_q.state_dict())
		# Disable updates for old network
		for param in self.target_critic_network_q.parameters():
			param.requires_grad_(False)

	
	def get_q_values(self, state_allies, state_enemies, one_hot_actions, rnn_hidden_state_q, indiv_dones):
		with torch.no_grad():
			indiv_masks = [1-d for d in indiv_dones]
			indiv_masks = torch.FloatTensor(indiv_masks).unsqueeze(0).unsqueeze(0)
			state_allies = torch.FloatTensor(state_allies).unsqueeze(0).unsqueeze(0)
			state_enemies = torch.FloatTensor(state_enemies).unsqueeze(0).unsqueeze(0)
			one_hot_actions = torch.FloatTensor(one_hot_actions).unsqueeze(0).unsqueeze(0)
			rnn_hidden_state_q = torch.FloatTensor(rnn_hidden_state_q)
			Q_value, weights_prd, _, rnn_hidden_state_q = self.target_critic_network_q(state_allies.to(self.device), state_enemies.to(self.device), one_hot_actions.to(self.device), rnn_hidden_state_q.to(self.device), indiv_masks.to(self.device))

			return Q_value.squeeze(0).cpu().numpy(), rnn_hidden_state_q.cpu().numpy()

	
	def get_action(self, state_policy, last_one_hot_actions, mask_actions, hidden_state, greedy=False):
		with torch.no_grad():
			state_policy = torch.FloatTensor(state_policy).unsqueeze(0).unsqueeze(1).to(self.device)
			last_one_hot_actions = torch.FloatTensor(last_one_hot_actions).unsqueeze(0).unsqueeze(1).to(self.device)
			mask_actions = torch.BoolTensor(mask_actions).unsqueeze(0).unsqueeze(1).to(self.device)
			hidden_state = torch.FloatTensor(hidden_state).to(self.device)
			
			dists = []
			hidden_states = []
			for agent_id in range(self.num_agents):
				dists_i, hidden_state_i = self.policy_network[agent_id](state_policy[:, :, agent_id, :].unsqueeze(2), last_one_hot_actions[:, :, agent_id, :].unsqueeze(2), hidden_state[:, agent_id, :].unsqueeze(1), mask_actions[:, :, agent_id, :].unsqueeze(2))
				dists.append(dists_i)
				hidden_states.append(hidden_state_i)
			dists = torch.stack(dists, dim=0).squeeze(-2).permute(1, 2, 0, 3).float()
			hidden_states = torch.stack(hidden_states, dim=0).squeeze(-2).permute(1, 0, 2).float()

			if greedy:
				actions = [dist.argmax().detach().cpu().item() for dist in dists.reshape(self.num_agents, -1)]
				action_logprob = None
			else:
				actions = [Categorical(dist).sample().detach().cpu().item() for dist in dists.reshape(self.num_agents, -1)]

				probs = Categorical(dists)
				action_logprob = probs.log_prob(torch.FloatTensor(actions).to(self.device)).cpu().numpy()

			return actions, action_logprob, hidden_states.cpu().numpy()

	
	def plot(self, masks, episode):
		self.comet_ml.log_metric('Q_Value_Loss',self.plotting_dict["q_value_loss"],episode)
		self.comet_ml.log_metric('Grad_Norm_Q_Value',self.plotting_dict["grad_norm_value_q"],episode)
		self.comet_ml.log_metric('Policy_Loss',self.plotting_dict["policy_loss"],episode)
		self.comet_ml.log_metric('Grad_Norm_Policy',self.plotting_dict["grad_norm_policy"],episode)
		self.comet_ml.log_metric('Entropy',self.plotting_dict["entropy"],episode)

		# ENTROPY OF Q WEIGHTS
		for i in range(self.num_heads):
			entropy_weights = -torch.sum(torch.sum((self.plotting_dict["attention_weights_q"][:, i] * torch.log(torch.clamp(self.plotting_dict["attention_weights_q"][:, i], 1e-10, 1.0)) * masks.view(-1, self.num_agents, 1)), dim=-1))/masks.sum()
			self.comet_ml.log_metric('Q_Weight_Entropy_Head_'+str(i+1), entropy_weights.item(), episode)

	
	def update_parameters(self):
		if self.critic_weight_entropy_pen_final + self.critic_weight_entropy_pen_decay_rate > self.critic_weight_entropy_pen:
			self.critic_weight_entropy_pen += self.critic_weight_entropy_pen_decay_rate 

		if self.entropy_pen - self.entropy_pen_decay > self.entropy_pen_final:
			self.entropy_pen -= self.entropy_pen_decay


	def update(self, episode):
		# update reward model is being called from train_agent
		# for i in range(10):
		# 	self.update_reward_model(episode-(10-i))
		
		q_value_loss_batch = 0
		policy_loss_batch = 0
		entropy_batch = 0
		attention_weights_q_batch = None
		grad_norm_value_q_batch = 0
		grad_norm_policy_batch = 0

		self.buffer.calculate_targets(episode)

		agent_permutation = torch.randperm(self.num_agents)
		
		# torch.autograd.set_detect_anomaly(True)
		# Optimize policy for n epochs
		for _ in range(self.n_epochs):

			# SAMPLE DATA FROM BUFFER
			states_critic_allies, states_critic_enemies, hidden_state_q, states_actor, hidden_state_actor, logprobs_old, \
			actions, last_one_hot_actions, one_hot_actions, action_masks, masks, q_values_old, target_q_values, advantage  = self.buffer.sample_recurrent_policy()

			q_values_old *= masks

			target_shape = q_values_old.shape
			q_values, attention_weights_q, score_q, _ = self.critic_network_q(
												states_critic_allies.to(self.device),
												states_critic_enemies.to(self.device),
												one_hot_actions.to(self.device),
												hidden_state_q.to(self.device),
												masks.to(self.device),
												)
			q_values = q_values.reshape(*target_shape)

			q_values *= masks.to(self.device)	
			target_q_values *= masks

			critic_q_loss_1 = F.huber_loss(q_values, target_q_values.to(self.device), reduction="sum", delta=10.0) / masks.sum()
			critic_q_loss_2 = F.huber_loss(torch.clamp(q_values, q_values_old.to(self.device)-self.value_clip, q_values_old.to(self.device)+self.value_clip), target_q_values.to(self.device), reduction="sum", delta=10.0) / masks.sum()

			entropy_weights = 0
			score_q_cum = 0
			for i in range(self.num_heads):
				entropy_weights += -torch.sum(torch.sum((attention_weights_q[:, i] * torch.log(torch.clamp(attention_weights_q[:, i], 1e-10, 1.0)) * masks.view(-1, self.num_agents, 1).to(self.device)), dim=-1))/masks.sum()
				score_q_cum += (score_q[:, i].squeeze(-2)**2 * masks.view(-1, self.num_agents, 1).to(self.device)).sum()/masks.sum()
			
			critic_q_loss = torch.max(critic_q_loss_1, critic_q_loss_2) + self.critic_score_regularizer*score_q_cum + self.critic_weight_entropy_pen*entropy_weights

			self.q_critic_optimizer.zero_grad()
			critic_q_loss.backward()
			if self.enable_grad_clip_critic_q:
				grad_norm_value_q = torch.nn.utils.clip_grad_norm_(self.critic_network_q.parameters(), self.grad_clip_critic_q)
			else:
				total_norm = 0
				for p in self.critic_network_q.parameters():
					param_norm = p.grad.detach().data.norm(2)
					total_norm += param_norm.item() ** 2
				grad_norm_value_q = torch.tensor([total_norm ** 0.5])
			self.q_critic_optimizer.step()

			if self.norm_adv:
				shape = advantage.shape
				advantage_copy = copy.deepcopy(advantage)
				advantage_copy[masks.view(*shape) == 0.0] = float('nan')
				advantage_mean = torch.nanmean(advantage_copy)
				advantage_std = torch.from_numpy(np.array(np.nanstd(advantage_copy.cpu().numpy()))).float()
				
				advantage = ((advantage - advantage_mean) / (advantage_std + 1e-5))*masks.view(*shape)


			factor = torch.ones((self.max_time_steps, self.update_ppo_agent)).float()
			shape = hidden_state_actor.shape
			hidden_state_actor = hidden_state_actor.reshape(shape[0], self.max_time_steps, self.num_agents, -1)
			
			policy_loss_collect = 0
			policy_grad_norm_collect = 0
			policy_entropy_collect = 0

			for agent_id in agent_permutation:
				dists, _ = self.policy_network[agent_id](
							states_actor[:, :, agent_id, :].to(self.device).unsqueeze(-2),
							last_one_hot_actions[:, :, agent_id, :].to(self.device).unsqueeze(-2),
							hidden_state_actor[:, :, agent_id, :].to(self.device).unsqueeze(-2),
							action_masks[:, :, agent_id, :].to(self.device).unsqueeze(-2),
							)

				probs = Categorical(dists.squeeze(-2))
				logprobs = probs.log_prob(actions[:, :, agent_id].to(self.device))

				ratios = torch.exp((logprobs - logprobs_old[:, :, agent_id].to(self.device)))

				# Finding Surrogate Loss
				surr1 = ratios * advantage[:, :, agent_id].to(self.device) * masks[:, :, agent_id].to(self.device)
				surr2 = torch.clamp(ratios, 1-self.policy_clip, 1+self.policy_clip) * advantage[:, :, agent_id].to(self.device) * masks[:, :, agent_id].to(self.device)

				# final loss of clipped objective PPO
				entropy = -torch.sum(torch.sum(dists.squeeze(-2)*masks[:, :, agent_id].unsqueeze(-1).to(self.device) * torch.log(torch.clamp(dists.squeeze(-2)*masks[:, :, agent_id].unsqueeze(-1).to(self.device), 1e-10,1.0)), dim=-1))/ masks[:, :, agent_id].sum()
				policy_loss_ = ((-torch.min(surr1, surr2)*factor*masks[:, :, agent_id]).sum())/masks[:, :, agent_id].sum()
				policy_loss = policy_loss_ - self.entropy_pen*entropy

				self.policy_optimizer[agent_id].zero_grad()
				policy_loss.backward()
				if self.enable_grad_clip_actor:
					grad_norm_policy = torch.nn.utils.clip_grad_norm_(self.policy_network[agent_id].parameters(), self.grad_clip_actor)
				else:
					total_norm = 0
					for p in self.policy_network[agent_id].parameters():
						param_norm = p.grad.detach().data.norm(2)
						total_norm += param_norm.item() ** 2
					grad_norm_policy = torch.tensor([total_norm ** 0.5])
				self.policy_optimizer[agent_id].step()

				factor = factor*torch.prod(torch.exp(logprobs.unsqueeze(-1)-logprobs_old[:, :, agent_id].unsqueeze(-1).to(self.device)), dim=-1).reshape(self.max_time_steps, self.update_ppo_agent).detach()

				policy_loss_collect += policy_loss.item()
				policy_entropy_collect += entropy.item()
				policy_grad_norm_collect += grad_norm_policy.item()

				# del dists, probs, logprobs, ratios, surr1, surr2, entropy, policy_loss_, policy_loss

			policy_loss_collect /= self.num_agents
			policy_entropy_collect /= self.num_agents
			policy_grad_norm_collect /= self.num_agents

			q_value_loss_batch += critic_q_loss.item()
			policy_loss_batch += policy_loss_collect
			entropy_batch += policy_entropy_collect
			grad_norm_value_q_batch += grad_norm_value_q
			grad_norm_policy_batch += policy_grad_norm_collect
			if attention_weights_q_batch is None:
				attention_weights_q_batch = attention_weights_q.detach().cpu()
			else:
				attention_weights_q_batch += attention_weights_q.detach().cpu()
			

		# Copy new weights into old critic
		if self.soft_update_q:
			for target_param, param in zip(self.target_critic_network_q.parameters(), self.critic_network_q.parameters()):
				target_param.data.copy_(target_param.data * (1.0 - self.tau_q) + param.data * self.tau_q)
		elif episode % self.network_update_interval_q == 0:
			self.target_critic_network_q.load_state_dict(self.critic_network_q.state_dict())

		# clear buffer
		self.buffer.clear()
		

		q_value_loss_batch /= self.n_epochs
		policy_loss_batch /= self.n_epochs
		entropy_batch /= self.n_epochs
		grad_norm_value_q_batch /= self.n_epochs
		grad_norm_policy_batch /= self.n_epochs
		attention_weights_q_batch /= self.n_epochs


		
		self.plotting_dict = {
		"q_value_loss": q_value_loss_batch,
		"policy_loss": policy_loss_batch,
		"entropy": entropy_batch,
		"grad_norm_value_q": grad_norm_value_q_batch,
		"grad_norm_policy": grad_norm_policy_batch,
		"attention_weights_q": attention_weights_q_batch,
		}

		if self.comet_ml is not None:
			self.plot(masks, episode)

		del q_value_loss_batch, policy_loss_batch, entropy_batch, grad_norm_value_q_batch, grad_norm_policy_batch, attention_weights_q_batch
		torch.cuda.empty_cache()
