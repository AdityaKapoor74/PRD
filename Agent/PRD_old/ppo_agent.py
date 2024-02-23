import numpy as np
import time
import copy
import torch
import torch.optim as optim
from torch.distributions import Categorical
from ppo_model import *
import torch.nn.functional as F
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
		self.test_num = dictionary["test_num"]
		self.gif = dictionary["gif"]
		self.experiment_type = dictionary["experiment_type"]
		self.n_epochs = dictionary["n_epochs"]
		self.scheduler_need = dictionary["scheduler_need"]
		if dictionary["device"] == "gpu":
			self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		else:
			self.device = "cpu"

		self.update_ppo_agent = dictionary["update_ppo_agent"]
		self.max_time_steps = dictionary["max_time_steps"]
		
		# Critic Setup
		self.rnn_hidden_critic_dim = dictionary["rnn_hidden_critic_dim"]
		self.rnn_num_layers_critic = dictionary["rnn_num_layers_critic"]
		self.attention_drop_prob = dictionary["attention_drop_prob"]
		self.critic_ally_observation = dictionary["ally_observation"]
		self.critic_enemy_observation = dictionary["enemy_observation"]
		self.value_lr = dictionary["value_lr"]
		self.value_weight_decay = dictionary["value_weight_decay"]
		self.critic_weight_entropy_pen = dictionary["critic_weight_entropy_pen"]
		self.critic_score_regularizer = dictionary["critic_score_regularizer"]
		self.td_lambda = dictionary["td_lambda"] # TD lambda
		self.value_clip = dictionary["value_clip"]
		self.num_heads = dictionary["num_heads"]
		self.enable_grad_clip_critic = dictionary["enable_grad_clip_critic"]
		self.grad_clip_critic = dictionary["grad_clip_critic"]


		# Actor Setup
		self.rnn_hidden_actor_dim = dictionary["rnn_hidden_actor_dim"]
		self.rnn_num_layers_actor = dictionary["rnn_num_layers_actor"]
		self.data_chunk_length = dictionary["data_chunk_length"]
		self.actor_observation_shape = dictionary["local_observation"]
		self.policy_lr = dictionary["policy_lr"]
		self.policy_weight_decay = dictionary["policy_weight_decay"]
		self.update_learning_rate_with_prd = dictionary["update_learning_rate_with_prd"]
		self.gamma = dictionary["gamma"]
		self.entropy_pen = dictionary["entropy_pen"]
		self.gae_lambda = dictionary["gae_lambda"]
		self.top_k = dictionary["top_k"]
		self.norm_adv = dictionary["norm_adv"]
		self.norm_returns = dictionary["norm_returns"]
		self.threshold_min = dictionary["threshold_min"]
		self.threshold_max = dictionary["threshold_max"]
		self.steps_to_take = dictionary["steps_to_take"]
		self.policy_clip = dictionary["policy_clip"]
		self.enable_grad_clip_actor = dictionary["enable_grad_clip_actor"]
		self.grad_clip_actor = dictionary["grad_clip_actor"]
		
		self.select_above_threshold = dictionary["select_above_threshold"]
		self.final_select_above_threshold = self.select_above_threshold
		if "prd_above_threshold_decay" in self.experiment_type:
			self.threshold_delta = (self.select_above_threshold - self.threshold_min)/self.steps_to_take
		elif "prd_above_threshold_ascend" in self.experiment_type:
			self.threshold_delta = (self.threshold_max - self.select_above_threshold)/self.steps_to_take

		print("EXPERIMENT TYPE", self.experiment_type)

		self.critic_network = TransformerCritic(
			ally_obs_input_dim=self.critic_ally_observation, 
			enemy_obs_input_dim=self.critic_enemy_observation, 
			num_heads=self.num_heads, 
			num_agents=self.num_agents, 
			num_enemies=self.num_enemies, 
			num_actions=self.num_actions,
			rnn_num_layers=self.rnn_num_layers_critic,
			norm_output=dictionary["norm_returns_critic"],
			device=self.device, 
			attention_drop_prob=dictionary["attention_drop_prob"]
			).to(self.device)
		
		self.policy_network = Policy(
			obs_input_dim=self.actor_observation_shape+self.num_actions, 
			num_agents=self.num_agents, 
			num_actions=self.num_actions, 
			rnn_num_layers=self.rnn_num_layers_actor,
			device=self.device
			).to(self.device)


		self.buffer = RolloutBuffer(
			num_episodes=self.update_ppo_agent, 
			max_time_steps=self.max_time_steps, 
			num_agents=self.num_agents, 
			num_enemies=self.num_enemies,
			obs_shape_critic_ally=self.critic_ally_observation, 
			obs_shape_critic_enemy=self.critic_enemy_observation, 
			obs_shape_actor=self.actor_observation_shape, 
			rnn_num_layers_actor=self.rnn_num_layers_actor,
			actor_hidden_state_dim=self.rnn_hidden_actor_dim,
			rnn_num_layers_v=self.rnn_num_layers_critic,
			v_hidden_state_dim=self.rnn_hidden_critic_dim,
			num_actions=self.num_actions,
			data_chunk_length=self.data_chunk_length,
			norm_returns_v=dictionary["norm_returns_critic"],
			clamp_rewards=dictionary["clamp_rewards"],
			clamp_rewards_value_min=dictionary["clamp_rewards_value_min"],
			clamp_rewards_value_max=dictionary["clamp_rewards_value_max"],
			norm_rewards=dictionary["norm_rewards"],
			target_calc_style=dictionary["target_calc_style"],
			td_lambda=self.td_lambda,
			gae_lambda=self.gae_lambda,
			n_steps=dictionary["n_steps"],
			gamma=self.gamma,
			V_PopArt=self.critic_network.v_value_layer[-1],
			)

		
		if dictionary["load_models"]:
			# Loading models
			if torch.cuda.is_available() is False:
				# For CPU
				self.critic_network.load_state_dict(torch.load(dictionary["model_path_value"],map_location=torch.device('cpu')))
				self.policy_network.load_state_dict(torch.load(dictionary["model_path_policy"],map_location=torch.device('cpu')))
			else:
				# For GPU
				self.critic_network.load_state_dict(torch.load(dictionary["model_path_value"]))
				self.policy_network.load_state_dict(torch.load(dictionary["model_path_policy"]))

		
		self.critic_optimizer = optim.AdamW(self.critic_network.parameters(),lr=self.value_lr, weight_decay=self.value_weight_decay, eps=1e-5)
		self.policy_optimizer = optim.AdamW(self.policy_network.parameters(),lr=self.policy_lr, weight_decay=self.policy_weight_decay, eps=1e-5)

		if self.scheduler_need:
			self.scheduler_policy = optim.lr_scheduler.MultiStepLR(self.policy_optimizer, milestones=[1000, 20000], gamma=0.1)
			self.scheduler_value = optim.lr_scheduler.MultiStepLR(self.critic_optimizer, milestones=[1000, 20000], gamma=0.1)


		self.comet_ml = None
		if dictionary["save_comet_ml_plot"]:
			self.comet_ml = comet_ml


	def get_critic_output(self, state_allies, state_enemies, action_probs, one_hot_actions, rnn_hidden_state, indiv_dones):
		with torch.no_grad():
			indiv_masks = [1-d for d in indiv_dones]
			indiv_masks = torch.FloatTensor(indiv_masks).unsqueeze(0).unsqueeze(0)
			state_allies = torch.FloatTensor(state_allies).unsqueeze(0).unsqueeze(0)
			state_enemies = torch.FloatTensor(state_enemies).unsqueeze(0).unsqueeze(0)
			action_probs = torch.FloatTensor(action_probs).unsqueeze(0).unsqueeze(0)
			one_hot_actions = torch.FloatTensor(one_hot_actions).unsqueeze(0).unsqueeze(0)
			rnn_hidden_state = torch.FloatTensor(rnn_hidden_state)
			value, weights_prd, _, rnn_hidden_state = self.critic_network(state_allies.to(self.device), state_enemies.to(self.device), action_probs.to(self.device), one_hot_actions.to(self.device), rnn_hidden_state.to(self.device), indiv_masks.to(self.device))

			return value.squeeze(0).squeeze(-1).cpu().numpy(), rnn_hidden_state.cpu().numpy(), torch.mean(weights_prd.transpose(-1, -2), dim=1).cpu().numpy()




	def get_action(self, state_policy, last_one_hot_actions, mask_actions, hidden_state, greedy=False):
		with torch.no_grad():
			state_policy = torch.FloatTensor(state_policy).unsqueeze(0).unsqueeze(1)
			last_one_hot_actions = torch.FloatTensor(last_one_hot_actions).unsqueeze(0).unsqueeze(1)
			final_state_policy = torch.cat([state_policy, last_one_hot_actions], dim=-1).to(self.device)
			mask_actions = torch.BoolTensor(mask_actions).unsqueeze(0).unsqueeze(1).to(self.device)
			hidden_state = torch.FloatTensor(hidden_state).to(self.device)
			dists, hidden_state = self.policy_network(final_state_policy, hidden_state, mask_actions)
			if greedy:
				actions = [dist.argmax().detach().cpu().item() for dist in dists]
			else:
				actions = [Categorical(dist).sample().detach().cpu().item() for dist in dists.squeeze(0).squeeze(0)]

				probs = Categorical(dists)
				action_logprob = probs.log_prob(torch.FloatTensor(actions).to(self.device))

			return actions, action_logprob.squeeze(0).squeeze(0).cpu().numpy(), hidden_state.cpu().numpy(), dists.squeeze(0).squeeze(0).cpu().numpy()


	# def calculate_advantages(self, values, values_old, rewards, dones, masks_):
	# 	values = values.reshape(self.update_ppo_agent, -1, self.num_agents)
	# 	values_old = values.reshape(self.update_ppo_agent, -1, self.num_agents)
	# 	rewards = rewards.reshape(self.update_ppo_agent, -1, self.num_agents)
	# 	dones = dones.reshape(self.update_ppo_agent, -1, self.num_agents)
	# 	masks_ = masks_.reshape(self.update_ppo_agent, -1, 1)
	# 	advantages = rewards.new_zeros(*rewards.shape)
	# 	next_value = 0
	# 	advantage = 0
	# 	masks = 1 - dones
	# 	# counter = 0
	# 	for t in reversed(range(0, rewards.shape[1])):
	# 		# next_value = values_old.data[:,t+1,:]
	# 		td_error = rewards[:,t,:] + (self.gamma * next_value * masks[:,t,:]) - values.data[:,t,:]
	# 		next_value = values.data[:,t,:]
	# 		advantage = (td_error + (self.gamma * self.gae_lambda * advantage * masks[:,t,:]))*masks_[:,t,:]
	# 		# advantages.insert(0, advantage)
	# 		advantages[:,t,:] = advantage
	# 		# counter += 1

	# 	# advantages = torch.stack(advantages)
		
	# 	if self.norm_adv:
	# 		advantages = (advantages - advantages.mean()) / advantages.std()
		
	# 	return advantages.reshape(-1, self.num_agents)



	# def build_td_lambda_targets(self, rewards, terminated, mask, target_qs):
	# 	# Assumes  <target_qs > in B*T*A and <reward >, <terminated >  in B*T*A, <mask > in (at least) B*T-1*1
	# 	# Initialise  last  lambda -return  for  not  terminated  episodes
	# 	ret = target_qs.new_zeros(*target_qs.shape)
	# 	ret = target_qs * (1-terminated) # some episodes end early so we can't assume that by copying the last target_qs in ret would be good enough
	# 	# ret[:, -1] = target_qs[:, -1] * (1 - (torch.sum(terminated, dim=1)>0).int())
	# 	# Backwards  recursive  update  of the "forward  view"
	# 	for t in range(ret.shape[1] - 2, -1, -1):
	# 		ret[:, t] = self.lambda_ * self.gamma * ret[:, t + 1] + mask[:, t].unsqueeze(-1) \
	# 					* (rewards[:, t] + (1 - self.lambda_) * self.gamma * target_qs[:, t + 1] * (1 - terminated[:, t+1]))
	# 	# Returns lambda-return from t=0 to t=T-1, i.e. in B*T-1*A
	# 	# return ret[:, 0:-1]
	# 	return ret


	
	def plot(self, masks, episode):
		self.comet_ml.log_metric('V_Value_Loss',self.plotting_dict["v_value_loss"],episode)
		self.comet_ml.log_metric('Grad_Norm_V_Value',self.plotting_dict["grad_norm_value_v"],episode)
		self.comet_ml.log_metric('Policy_Loss',self.plotting_dict["policy_loss"],episode)
		self.comet_ml.log_metric('Grad_Norm_Policy',self.plotting_dict["grad_norm_policy"],episode)
		self.comet_ml.log_metric('Entropy',self.plotting_dict["entropy"],episode)

		if "threshold" in self.experiment_type:
			for i in range(self.num_agents):
				agent_name = "agent"+str(i)
				self.comet_ml.log_metric('Group_Size_'+agent_name, self.plotting_dict["agent_groups_over_episode"][i].item(), episode)

			self.comet_ml.log_metric('Avg_Group_Size', self.plotting_dict["avg_agent_group_over_episode"].item(), episode)

		# ENTROPY OF Q WEIGHTS
		for i in range(self.num_heads):
			entropy_weights = -torch.sum(torch.sum((self.plotting_dict["weights_prd"][:, i] * torch.log(torch.clamp(self.plotting_dict["weights_prd"][:, i], 1e-10, 1.0)) * masks.view(-1, self.num_agents, 1)), dim=-1))/masks.sum()
			self.comet_ml.log_metric('Q_Weight_Entropy_Head_'+str(i+1), entropy_weights.item(), episode)


	# def calculate_advantages_based_on_exp(self, V_values, V_values_old, rewards, dones, weights_prd, masks, episode):
	# 	advantage = None
	# 	masking_rewards = None
	# 	mean_min_weight_value = -1
	# 	if "shared" in self.experiment_type:
	# 		rewards_ = torch.sum(rewards.unsqueeze(-2).repeat(1, self.num_agents, 1), dim=-1)
	# 		advantage = self.calculate_advantages(V_values, V_values_old, rewards_, dones, masks)
	# 	elif "prd_above_threshold_ascend" in self.experiment_type or "prd_above_threshold_decay" in self.experiment_type:
	# 		masking_rewards = (weights_prd>self.select_above_threshold).int()
	# 		rewards_ = torch.sum(rewards.unsqueeze(-2).repeat(1, self.num_agents, 1) * torch.transpose(masking_rewards,-1,-2), dim=-1)
	# 		advantage = self.calculate_advantages(V_values, V_values_old, rewards_, dones, masks)
	# 	elif "prd_above_threshold" in self.experiment_type:
	# 		if episode > self.steps_to_take:
	# 			masking_rewards = (weights_prd>self.select_above_threshold).int()
	# 			rewards_ = torch.sum(rewards.unsqueeze(-2).repeat(1, self.num_agents, 1) * torch.transpose(masking_rewards,-1,-2), dim=-1)
	# 		else:
	# 			masking_rewards = torch.ones(weights_prd.shape).to(self.device)
	# 			rewards_ = torch.sum(rewards.unsqueeze(-2).repeat(1, self.num_agents, 1), dim=-1)
	# 		advantage = self.calculate_advantages(V_values, V_values_old, rewards_, dones, masks)
	# 	elif "top" in self.experiment_type:
	# 		if episode > self.steps_to_take:
	# 			rewards_ = torch.sum(rewards.unsqueeze(-2).repeat(1, self.num_agents, 1), dim=-1)
	# 			advantage = self.calculate_advantages(V_values, V_values_old, rewards_, dones, masks)
	# 			masking_rewards = torch.ones(weights_prd.shape).to(self.device)
	# 			min_weight_values, _ = torch.min(weights_prd, dim=-1)
	# 			mean_min_weight_value = torch.mean(min_weight_values)
	# 		else:
	# 			values, indices = torch.topk(weights_prd,k=self.top_k,dim=-1)
	# 			min_weight_values, _ = torch.min(values, dim=-1)
	# 			mean_min_weight_value = torch.mean(min_weight_values)
	# 			masking_rewards = torch.sum(F.one_hot(indices, num_classes=self.num_agents), dim=-2)
	# 			rewards_ = torch.sum(masking_rewards * torch.transpose(masking_rewards,-1,-2), dim=-1)
	# 			advantage = self.calculate_advantages(V_values, V_values_old, rewards_, dones, masks)
	# 	elif "prd_soft_advantage" in self.experiment_type:
	# 		if episode > self.steps_to_take:
	# 			rewards_ = torch.sum(rewards.unsqueeze(-2).repeat(1, self.num_agents, 1) * torch.transpose(weights_prd * self.num_agents, -1, -2), dim=-1)
	# 		else:
	# 			rewards_ = torch.sum(rewards.unsqueeze(-2).repeat(1, self.num_agents, 1), dim=-1)
	# 		advantage = self.calculate_advantages(V_values, V_values_old, rewards_, dones, masks)
		
	# 	if "scaled" in self.experiment_type and episode > self.steps_to_take and "top" in self.experiment_type:
	# 		advantage *= self.num_agents
	# 		if "top" in self.experiment_type:
	# 			advantage /= self.top_k


	# 	return advantage.detach(), masking_rewards, mean_min_weight_value


	def update_parameters(self, episode):

		if episode>self.steps_to_take and self.experiment_type == "prd_above_threshold":
			self.select_above_threshold = self.final_select_above_threshold
		else:
			self.select_above_threshold = 0.0

		if self.select_above_threshold > self.threshold_min and "prd_above_threshold_decay" in self.experiment_type:
			self.select_above_threshold = self.select_above_threshold - self.threshold_delta

		if self.threshold_max >= self.select_above_threshold and "prd_above_threshold_ascend" in self.experiment_type:
			self.select_above_threshold = self.select_above_threshold + self.threshold_delta


	def update(self, episode):
		
		v_value_loss_batch = 0
		policy_loss_batch = 0
		entropy_batch = 0
		weight_v_batch = None
		grad_norm_value_v_batch = 0
		grad_norm_policy_batch = 0
		agent_groups_over_episode_batch = 0
		avg_agent_group_over_episode_batch = 0

		self.buffer.calculate_targets(self.experiment_type, episode, self.select_above_threshold)

		
		# torch.autograd.set_detect_anomaly(True)
		# Optimize policy for n epochs
		for _ in range(self.n_epochs):

			# SAMPLE DATA FROM BUFFER
			states_critic_allies, states_critic_enemies, hidden_state_v, states_actor, hidden_state_actor, logprobs_old, \
			actions, action_probs, last_one_hot_actions, one_hot_actions, action_masks, masks, values_old, target_values, advantage  = self.buffer.sample_recurrent_policy()


			values_old *= masks.unsqueeze(-2).repeat(1, 1, self.num_agents, 1)

			if self.norm_adv:
				shape = advantage.shape
				advantage_copy = copy.deepcopy(advantage)
				advantage_copy[masks.view(*shape) == 0.0] = float('nan')
				advantage_mean = torch.nanmean(advantage_copy)
				advantage_std = torch.from_numpy(np.array(np.nanstd(advantage_copy.cpu().numpy()))).float()
				advantage = ((advantage - advantage_mean) / (advantage_std + 1e-5))*masks.view(*shape)


			target_shape = values_old.shape
			values, weight_v, score_v, h_v = self.critic_network(
												states_critic_allies.to(self.device),
												states_critic_enemies.to(self.device),
												action_probs.to(self.device),
												one_hot_actions.to(self.device),
												hidden_state_v.to(self.device),
												masks.to(self.device),
												)
			values = values.reshape(*target_shape)

			dists, _ = self.policy_network(
					torch.cat([states_actor, last_one_hot_actions], dim=-1).to(self.device),
					hidden_state_actor.to(self.device),
					action_masks.to(self.device),
					)

			values *= masks.unsqueeze(-2).repeat(1, 1, self.num_agents, 1).to(self.device)
			target_values *= masks.unsqueeze(-2).repeat(1, 1, self.num_agents, 1)

			probs = Categorical(dists)
			logprobs = probs.log_prob(actions.to(self.device))
			
			if "threshold" in self.experiment_type or "top" in self.experiment_type:
				mask_rewards = (weight_v.transpose(-1, -2)>self.select_above_threshold).int()
				agent_groups_over_episode = torch.sum(torch.sum(mask_rewards.reshape(-1, self.num_agents, self.num_agents).float(), dim=-2),dim=0)/mask_rewards.reshape(-1, self.num_agents, self.num_agents).shape[0]
				avg_agent_group_over_episode = torch.mean(agent_groups_over_episode)
				agent_groups_over_episode_batch += agent_groups_over_episode
				avg_agent_group_over_episode_batch += avg_agent_group_over_episode
				

			critic_v_loss_1 = F.huber_loss(values, target_values.to(self.device), reduction="sum", delta=10.0) / masks.sum()
			critic_v_loss_2 = F.huber_loss(torch.clamp(values, values_old.to(self.device)-self.value_clip, values_old.to(self.device)+self.value_clip), target_values.to(self.device), reduction="sum", delta=10.0) / masks.sum()

			
			critic_v_loss = torch.max(critic_v_loss_1, critic_v_loss_2)


			# Finding the ratio (pi_theta / pi_theta__old)
			ratios = torch.exp((logprobs - logprobs_old.to(self.device)))
			
			# Finding Surrogate Loss
			surr1 = ratios * advantage.to(self.device) * masks.to(self.device)
			surr2 = torch.clamp(ratios, 1-self.policy_clip, 1+self.policy_clip) * advantage.to(self.device) * masks.to(self.device)

			# final loss of clipped objective PPO
			entropy = -torch.sum(torch.sum(dists*masks.unsqueeze(-1).to(self.device) * torch.log(torch.clamp(dists*masks.unsqueeze(-1).to(self.device), 1e-10,1.0)), dim=-1))/ masks.sum() #(masks.sum()*self.num_agents)
			policy_loss_ = (-torch.min(surr1, surr2).sum())/masks.sum()
			policy_loss = policy_loss_ - self.entropy_pen*entropy

			print("Policy Loss", policy_loss_.item(), "Entropy", (-self.entropy_pen*entropy.item()))

			self.critic_optimizer.zero_grad()
			critic_v_loss.backward()
			if self.enable_grad_clip_critic:
				grad_norm_value_v = torch.nn.utils.clip_grad_norm_(self.critic_network.parameters(), self.grad_clip_critic_v)
			else:
				total_norm = 0
				for p in self.critic_network.parameters():
					if p.grad is None:
						continue
					param_norm = p.grad.detach().data.norm(2)
					total_norm += param_norm.item() ** 2
				grad_norm_value_v = torch.tensor([total_norm ** 0.5])
			self.critic_optimizer.step()

			self.policy_optimizer.zero_grad()
			policy_loss.backward()
			if self.enable_grad_clip_actor:
				grad_norm_policy = torch.nn.utils.clip_grad_norm_(self.policy_network.parameters(), self.grad_clip_actor)
			else:
				total_norm = 0
				for p in self.policy_network.parameters():
					if p.grad is None:
						continue
					param_norm = p.grad.detach().data.norm(2)
					total_norm += param_norm.item() ** 2
				grad_norm_policy = torch.tensor([total_norm ** 0.5])
			self.policy_optimizer.step()

			v_value_loss_batch += critic_v_loss.item()
			policy_loss_batch += policy_loss.item()
			entropy_batch += entropy.item()
			grad_norm_value_v_batch += grad_norm_value_v
			grad_norm_policy_batch += grad_norm_policy
			if weight_v_batch is None:
				weight_v_batch = weight_v.detach().cpu()
			else:
				weight_v_batch += weight_v.detach().cpu()
			

		
		self.buffer.clear()
		
		v_value_loss_batch /= self.n_epochs
		policy_loss_batch /= self.n_epochs
		entropy_batch /= self.n_epochs
		grad_norm_value_v_batch /= self.n_epochs
		grad_norm_policy_batch /= self.n_epochs
		weight_v_batch /= self.n_epochs
		agent_groups_over_episode_batch /= self.n_epochs
		avg_agent_group_over_episode_batch /= self.n_epochs


		self.plotting_dict = {
		"v_value_loss": v_value_loss_batch,
		"policy_loss": policy_loss_batch,
		"entropy": entropy_batch,
		"grad_norm_value_v":grad_norm_value_v_batch,
		"grad_norm_policy": grad_norm_policy_batch,
		"weights_prd": weight_v_batch,
		}

		if "threshold" in self.experiment_type:
			self.plotting_dict["agent_groups_over_episode"] = agent_groups_over_episode_batch
			self.plotting_dict["avg_agent_group_over_episode"] = avg_agent_group_over_episode_batch

		if self.comet_ml is not None:
			self.plot(masks, episode)

		del v_value_loss_batch, policy_loss_batch, entropy_batch, grad_norm_value_v_batch, grad_norm_policy_batch, weight_v_batch, agent_groups_over_episode_batch, avg_agent_group_over_episode_batch
		torch.cuda.empty_cache()


	# def update(self, episode):

	# 	# convert list to tensor
	# 	# BATCH, TIMESTEPS, NUM AGENTS, DIM
	# 	old_states_critic_allies = torch.FloatTensor(np.array(self.buffer.states_critic_allies))
	# 	old_states_critic_enemies = torch.FloatTensor(np.array(self.buffer.states_critic_enemies))
	# 	# old_V_values = torch.FloatTensor(np.array(self.buffer.V_values))
	# 	# old_Q_values = torch.FloatTensor(np.array(self.buffer.Q_values))
	# 	# old_weights_prd = torch.FloatTensor(np.array(self.buffer.weights_prd))
	# 	old_states_actor = torch.FloatTensor(np.array(self.buffer.states_actor))
	# 	old_actions = torch.FloatTensor(np.array(self.buffer.actions)).reshape(-1, self.num_agents)
	# 	old_last_one_hot_actions = torch.FloatTensor(np.array(self.buffer.last_one_hot_actions))
	# 	old_one_hot_actions = torch.FloatTensor(np.array(self.buffer.one_hot_actions))
	# 	old_dists = torch.FloatTensor(np.array(self.buffer.dists))
	# 	old_mask_actions = torch.BoolTensor(np.array(self.buffer.action_masks)).reshape(-1, self.num_agents, self.num_actions)
	# 	old_logprobs = torch.FloatTensor(self.buffer.logprobs).reshape(-1, self.num_agents)
	# 	rewards = torch.FloatTensor(np.array(self.buffer.rewards)).reshape(-1, self.num_agents)
	# 	dones = torch.FloatTensor(np.array(self.buffer.dones)).long().reshape(-1, self.num_agents)
	# 	masks = torch.FloatTensor(np.array(self.buffer.masks)).long()

	# 	batch, time_steps = masks.shape
	# 	rnn_hidden_state_critic = torch.zeros(1, batch*self.num_agents*self.num_agents, self.rnn_hidden_critic)
	# 	rnn_hidden_state_actor = torch.zeros(1, batch*self.num_agents, self.rnn_hidden_actor)

	# 	with torch.no_grad():
	# 		V_values_old, _, _ = self.critic_network(old_states_critic_allies.to(self.device), old_states_critic_enemies.to(self.device), old_dists.to(self.device), old_one_hot_actions.to(self.device), rnn_hidden_state_critic.to(self.device))
	# 		V_values_old = V_values_old.reshape(-1, self.num_agents, self.num_agents)

	# 	shape = (batch, time_steps, self.num_agents)
	# 	target_V_values = self.build_td_lambda_targets(rewards.reshape(*shape).unsqueeze(-1).to(self.device), dones.reshape(*shape).unsqueeze(-1).to(self.device), masks.reshape(*shape[:-1]).unsqueeze(-1).to(self.device), V_values_old.reshape(self.update_ppo_agent, -1, self.num_agents, self.num_agents)).reshape(-1, self.num_agents, self.num_agents)

	# 	value_loss_batch = 0.0
	# 	policy_loss_batch = 0.0
	# 	entropy_batch = 0.0
	# 	grad_norm_value_batch = 0.0
	# 	grad_norm_policy_batch = 0.0
	# 	weights_value_batch = None

	# 	rewards = rewards.reshape(-1, self.num_agents)
	# 	dones = dones.reshape(-1, self.num_agents)
	# 	masks = masks.reshape(-1, 1)

	# 	for _ in range(self.n_epochs):

	# 		'''
	# 		Getting the probability mass function over the action space for each agent
	# 		'''
	# 		# start_forward_time = time.process_time()

	# 		dists, _ = self.policy_network(
	# 				torch.cat([old_states_actor, old_last_one_hot_actions], dim=-1).to(self.device),
	# 				rnn_hidden_state_actor.to(self.device),
	# 				old_mask_actions.to(self.device),
	# 				update=True
	# 				)
	# 		probs = Categorical(dists)
	# 		logprobs = probs.log_prob(old_actions.to(self.device))

	# 		'''
	# 		Calculate V values
	# 		'''
	# 		V_values, weights_value, _ = self.critic_network(old_states_critic_allies.to(self.device), old_states_critic_enemies.to(self.device), old_dists.to(self.device), old_one_hot_actions.to(self.device), rnn_hidden_state_critic.to(self.device))
	# 		V_values = V_values.reshape(-1,self.num_agents,self.num_agents)
		

	# 		# value_loss = self.calculate_value_loss(V_values, target_V_values, weights_value)
	# 		critic_v_loss_1 = F.mse_loss(V_values*masks.unsqueeze(-1).to(self.device), target_V_values*masks.unsqueeze(-1).to(self.device), reduction="sum") / masks.sum()
	# 		critic_v_loss_2 = F.mse_loss(torch.clamp(V_values, V_values_old.to(self.device)-self.value_clip, V_values_old.to(self.device)+self.value_clip)*masks.unsqueeze(-1).to(self.device), target_V_values*masks.unsqueeze(-1).to(self.device), reduction="sum") / masks.sum()

	# 		value_loss = torch.max(critic_v_loss_1, critic_v_loss_2)

	# 		advantage, masking_advantage, _ = self.calculate_advantages_based_on_exp(V_values, V_values, rewards.to(self.device), dones.to(self.device), torch.mean(weights_value.detach(), dim=1), masks.to(self.device), episode)

	# 		if "prd_avg" in self.experiment_type:
	# 			agent_groups_over_episode = torch.sum(masking_advantage,dim=0)
	# 			avg_agent_group_over_episode = torch.mean(agent_groups_over_episode.float())
	# 		elif "threshold" in self.experiment_type:
	# 			agent_groups_over_episode = torch.sum(torch.sum(masking_advantage.float(), dim=-2),dim=0)/masking_advantage.shape[0]
	# 			avg_agent_group_over_episode = torch.mean(agent_groups_over_episode)
			

	# 		# Finding the ratio (pi_theta / pi_theta__old)
	# 		ratios = torch.exp(logprobs - old_logprobs.to(self.device))
	# 		# Finding Surrogate Loss
	# 		surr1 = ratios * advantage * masks.to(self.device)
	# 		surr2 = torch.clamp(ratios, 1-self.policy_clip, 1+self.policy_clip) * advantage * masks.to(self.device)

	# 		# final loss of clipped objective PPO
	# 		# entropy = -torch.mean(torch.sum(dists*masks.unsqueeze(-1).to(self.device) * torch.log(torch.clamp(dists*masks.unsqueeze(-1).to(self.device), 1e-10,1.0)), dim=2))
	# 		entropy = -torch.sum(torch.sum(dists*masks.unsqueeze(-1).to(self.device) * torch.log(torch.clamp(dists*masks.unsqueeze(-1).to(self.device), 1e-10,1.0)), dim=-1))/(masks.sum()*self.num_agents)
	# 		policy_loss = ((-torch.min(surr1, surr2).sum())/(masks.sum()*self.num_agents) - self.entropy_pen*entropy)


	# 		self.critic_optimizer.zero_grad()
	# 		value_loss.backward(retain_graph=False)
	# 		if self.enable_grad_clip_critic:
	# 			grad_norm_value = torch.nn.utils.clip_grad_norm_(self.critic_network.parameters(),self.grad_clip_critic)
	# 		else:
	# 			grad_norm_value = torch.tensor([1.0])
	# 		# grad_norm_value = 0
	# 		# for p in self.critic_network.parameters():
	# 		# 	param_norm = p.grad.detach().data.norm(2)
	# 		# 	grad_norm_value += param_norm.item() ** 2
	# 		# grad_norm_value = torch.tensor(grad_norm_value) ** 0.5
	# 		self.critic_optimizer.step()

	# 		self.policy_optimizer.zero_grad()
	# 		policy_loss.backward(retain_graph=False)
	# 		if self.enable_grad_clip_actor:
	# 			grad_norm_policy = torch.nn.utils.clip_grad_norm_(self.policy_network.parameters(),self.grad_clip_actor)
	# 		else:
	# 			grad_norm_policy = torch.tensor([1.0])
	# 		self.policy_optimizer.step()

	# 		value_loss_batch += value_loss.item()
	# 		policy_loss_batch += policy_loss.item()
	# 		grad_norm_policy_batch += grad_norm_policy.item()
	# 		grad_norm_value_batch += grad_norm_value.item()
	# 		entropy_batch += entropy.item()
	# 		if weights_value_batch is None:
	# 			weights_value_batch = weights_value.detach()
	# 		else:
	# 			weights_value_batch += weights_value.detach()


	# 	value_loss_batch /= self.n_epochs
	# 	policy_loss_batch /= self.n_epochs
	# 	grad_norm_policy_batch /= self.n_epochs
	# 	grad_norm_value_batch /= self.n_epochs
	# 	entropy_batch /= self.n_epochs
	# 	weights_value_batch /= self.n_epochs


	# 	self.update_parameters()

	# 	# clear buffer
	# 	self.buffer.clear()


	# 	self.plotting_dict = {
	# 	"value_loss": value_loss_batch,
	# 	"policy_loss": policy_loss_batch,
	# 	"entropy": entropy_batch,
	# 	"grad_norm_value":grad_norm_value_batch,
	# 	"grad_norm_policy": grad_norm_policy_batch,
	# 	"weights_value": weights_value_batch,
	# 	}

	# 	if "threshold" in self.experiment_type:
	# 		self.plotting_dict["agent_groups_over_episode"] = agent_groups_over_episode
	# 		self.plotting_dict["avg_agent_group_over_episode"] = avg_agent_group_over_episode
	# 	if "prd_top" in self.experiment_type:
	# 		self.plotting_dict["mean_min_weight_value"] = mean_min_weight_value

	# 	if self.comet_ml is not None:
	# 		self.plot(episode)
