import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from model import MLP_Policy, Q_network, V_network
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
		self.temperature_v = dictionary["temperature_v"]
		self.temperature_q = dictionary["temperature_q"]
		self.rnn_hidden_q = dictionary["rnn_hidden_q"]
		self.rnn_hidden_v = dictionary["rnn_hidden_v"]
		self.critic_observation_shape = dictionary["global_observation"]
		self.q_value_lr = dictionary["q_value_lr"]
		self.v_value_lr = dictionary["v_value_lr"]
		self.q_weight_decay = dictionary["q_weight_decay"]
		self.v_weight_decay = dictionary["v_weight_decay"]
		self.critic_weight_entropy_pen = dictionary["critic_weight_entropy_pen"]
		self.critic_score_regularizer = dictionary["critic_score_regularizer"]
		self.lambda_ = dictionary["lambda"] # TD lambda
		self.value_clip = dictionary["value_clip"]
		self.num_heads = dictionary["num_heads"]
		self.enable_hard_attention = dictionary["enable_hard_attention"]
		self.enable_grad_clip_critic = dictionary["enable_grad_clip_critic"]
		self.grad_clip_critic = dictionary["grad_clip_critic"]


		# Actor Setup
		self.warm_up = dictionary["warm_up"]
		self.warm_up_episodes = dictionary["warm_up_episodes"]
		self.epsilon_start = self.epsilon = dictionary["epsilon_start"]
		self.epsilon_end = dictionary["epsilon_end"]
		self.rnn_hidden_actor = dictionary["rnn_hidden_actor"]
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
		if self.enable_hard_attention:
			self.select_above_threshold = 0
		else:
			self.select_above_threshold = dictionary["select_above_threshold"]
			if "prd_above_threshold_decay" in self.experiment_type:
				self.threshold_delta = (self.select_above_threshold - self.threshold_min)/self.steps_to_take
			elif "prd_above_threshold_ascend" in self.experiment_type:
				self.threshold_delta = (self.threshold_max - self.select_above_threshold)/self.steps_to_take


		print("EXPERIMENT TYPE", self.experiment_type)
		# obs_input_dim = 2*3+1 # crossing_team_greedy
		# Q-V Network
		self.critic_network_q = Q_network(obs_input_dim=self.critic_observation_shape, num_heads=self.num_heads, num_agents=self.num_agents, num_actions=self.num_actions, device=self.device, enable_hard_attention=self.enable_hard_attention, attention_dropout_prob=dictionary["attention_dropout_prob_q"], temperature=self.temperature_q).to(self.device)
		self.critic_network_q_old = Q_network(obs_input_dim=self.critic_observation_shape, num_heads=self.num_heads, num_agents=self.num_agents, num_actions=self.num_actions, device=self.device, enable_hard_attention=self.enable_hard_attention, attention_dropout_prob=dictionary["attention_dropout_prob_q"], temperature=self.temperature_q).to(self.device)
		# Copy network params
		self.critic_network_q_old.load_state_dict(self.critic_network_q.state_dict())
		# Disable updates for old network
		for param in self.critic_network_q_old.parameters():
			param.requires_grad_(False)

		self.critic_network_v = V_network(obs_input_dim=self.critic_observation_shape, num_heads=self.num_heads, num_agents=self.num_agents, num_actions=self.num_actions, device=self.device, enable_hard_attention=self.enable_hard_attention, attention_dropout_prob=dictionary["attention_dropout_prob_v"], temperature=self.temperature_v).to(self.device)
		self.critic_network_v_old = V_network(obs_input_dim=self.critic_observation_shape, num_heads=self.num_heads, num_agents=self.num_agents, num_actions=self.num_actions, device=self.device, enable_hard_attention=self.enable_hard_attention, attention_dropout_prob=dictionary["attention_dropout_prob_v"], temperature=self.temperature_v).to(self.device)
		# Copy network params
		self.critic_network_v_old.load_state_dict(self.critic_network_v.state_dict())
		# Disable updates for old network
		for param in self.critic_network_v_old.parameters():
			param.requires_grad_(False)
		
		
		# Policy Network
		# obs_input_dim = 2*3+1 + (self.num_agents-1)*(2*2+1) # crossing_team_greedy
		self.policy_network = MLP_Policy(obs_input_dim=self.actor_observation_shape, num_agents=self.num_agents, num_actions=self.num_actions, device=self.device).to(self.device)
		self.policy_network_old = MLP_Policy(obs_input_dim=self.actor_observation_shape, num_agents=self.num_agents, num_actions=self.num_actions, device=self.device).to(self.device)
		# Copy network params
		self.policy_network_old.load_state_dict(self.policy_network.state_dict())
		# Disable updates for old network
		for param in self.policy_network_old.parameters():
			param.requires_grad_(False)

		self.network_update_interval = dictionary["network_update_interval"]
		
		self.buffer = RolloutBuffer(
			num_episodes=self.update_ppo_agent, 
			max_time_steps=self.max_time_steps, 
			num_agents=self.num_agents, 
			obs_shape_critic=self.critic_observation_shape, 
			obs_shape_actor=self.actor_observation_shape, 
			num_actions=self.num_actions,
			rnn_hidden_actor=self.rnn_hidden_actor,
			rnn_hidden_q=self.rnn_hidden_q,
			rnn_hidden_v=self.rnn_hidden_v,
			)

		# Loading models
		if dictionary["load_models"]:
			# For CPU
			if torch.cuda.is_available() is False:
				self.critic_network.load_state_dict(torch.load(dictionary["model_path_value"], map_location=torch.device('cpu')))
				self.policy_network.load_state_dict(torch.load(dictionary["model_path_policy"], map_location=torch.device('cpu')))
			# For GPU
			else:
				self.critic_network.load_state_dict(torch.load(dictionary["model_path_value"]))
				self.policy_network.load_state_dict(torch.load(dictionary["model_path_policy"]))


		self.q_critic_optimizer = optim.Adam(self.critic_network_q.parameters(), lr=self.v_value_lr, weight_decay=self.v_weight_decay)
		self.v_critic_optimizer = optim.Adam(self.critic_network_v.parameters(), lr=self.q_value_lr, weight_decay=self.q_weight_decay)
		self.policy_optimizer = optim.Adam(self.policy_network.parameters(),lr=self.policy_lr, weight_decay=self.policy_weight_decay)

		if self.scheduler_need:
			self.scheduler_policy = optim.lr_scheduler.MultiStepLR(self.policy_optimizer, milestones=[1000, 20000], gamma=0.1)
			self.scheduler_q_critic = optim.lr_scheduler.MultiStepLR(self.q_critic_optimizer, milestones=[1000, 20000], gamma=0.1)
			self.scheduler_v_critic = optim.lr_scheduler.MultiStepLR(self.v_critic_optimizer, milestones=[1000, 20000], gamma=0.1)

		self.comet_ml = None
		if dictionary["save_comet_ml_plot"]:
			self.comet_ml = comet_ml

	def get_value_rnn_state(self, state, one_hot_actions):
		with torch.no_grad():
			state = torch.FloatTensor(state).unsqueeze(0)
			one_hot_actions = torch.FloatTensor(one_hot_actions).unsqueeze(0)
			_, _, _, rnn_hidden_state_v = self.critic_network_v_old(state.to(self.device), one_hot_actions.to(self.device))
			_, _, _, rnn_hidden_state_q = self.critic_network_q_old(state.to(self.device), one_hot_actions.to(self.device))

			return rnn_hidden_state_v.cpu().numpy(), rnn_hidden_state_q.cpu().numpy()

	def update_epsilon(self):
		if self.warm_up:
			self.epsilon -= (self.epsilon_start - self.epsilon_end)/self.warm_up_episodes
			self.epsilon = max(self.epsilon, self.epsilon_end)

	def get_action(self, state_policy, mask_actions, greedy=False):
		with torch.no_grad():
			state_policy = torch.FloatTensor(state_policy).to(self.device)
			mask_actions = torch.FloatTensor(mask_actions).to(self.device)
			dists, rnn_hidden_state = self.policy_network_old(state_policy, mask_actions)
			if self.warm_up:
				available_actions = (mask_actions>=0).int()
				dists = (1.0-self.epsilon)*dists + available_actions*self.epsilon/torch.sum(available_actions, dim=-1).unsqueeze(-1)
			if greedy:
				actions = [dist.argmax().detach().cpu().item() for dist in dists]
			else:
				actions = [Categorical(dist).sample().detach().cpu().item() for dist in dists]

				probs = Categorical(dists)
				action_logprob = probs.log_prob(torch.FloatTensor(actions).to(self.device))

			return actions, action_logprob.cpu().numpy(), rnn_hidden_state.cpu().numpy()


	def calculate_advantages(self, values, values_old, rewards, dones, masks_):
		advantages = []
		next_value = 0
		advantage = 0
		masks = 1 - dones
		for t in reversed(range(0, len(rewards))):
			td_error = rewards[t] + (self.gamma * next_value * masks[t]) - values.data[t]
			next_value = values_old.data[t]
			advantage = (td_error + (self.gamma * self.gae_lambda * advantage * masks[t]))*masks_[t]
			advantages.insert(0, advantage)

		advantages = torch.stack(advantages)
		
		if self.norm_adv:
			advantages = (advantages - advantages.mean()) / advantages.std()
		
		return advantages



	def build_td_lambda_targets(self, rewards, terminated, mask, target_qs):
		# Assumes  <target_qs > in B*T*A and <reward >, <terminated >  in B*T*A, <mask > in (at least) B*T-1*1
		# Initialise  last  lambda -return  for  not  terminated  episodes
		ret = target_qs.new_zeros(*target_qs.shape)
		ret = target_qs * (1-terminated)
		# ret[:, -1] = target_qs[:, -1] * (1 - (torch.sum(terminated, dim=1)>0).int())
		# Backwards  recursive  update  of the "forward  view"
		for t in range(ret.shape[1] - 2, -1, -1):
			ret[:, t] = self.lambda_ * self.gamma * ret[:, t + 1] + mask[:, t].unsqueeze(-1) \
						* (rewards[:, t] + (1 - self.lambda_) * self.gamma * target_qs[:, t + 1] * (1 - terminated[:, t]))
		# Returns lambda-return from t=0 to t=T-1, i.e. in B*T-1*A
		# return ret[:, 0:-1]
		return ret



	def plot(self, episode):
		self.comet_ml.log_metric('Q_Value_Loss',self.plotting_dict["q_value_loss"],episode)
		self.comet_ml.log_metric('V_Value_Loss',self.plotting_dict["v_value_loss"],episode)
		self.comet_ml.log_metric('Grad_Norm_V_Value',self.plotting_dict["grad_norm_value_v"],episode)
		self.comet_ml.log_metric('Grad_Norm_Q_Value',self.plotting_dict["grad_norm_value_q"],episode)
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
			entropy_weights = -torch.mean(torch.sum(self.plotting_dict["weights_prd"][:,i]* torch.log(torch.clamp(self.plotting_dict["weights_prd"][:,i], 1e-10,1.0)), dim=2))
			self.comet_ml.log_metric('Q_Weight_Entropy_Head_'+str(i+1), entropy_weights.item(), episode)

		# ENTROPY OF V WEIGHTS
		for i in range(self.num_heads):
			entropy_weights = -torch.mean(torch.sum(self.plotting_dict["weights_v"][:,i]* torch.log(torch.clamp(self.plotting_dict["weights_v"][:,i], 1e-10,1.0)), dim=2))
			self.comet_ml.log_metric('V_Weight_Entropy_Head_'+str(i+1), entropy_weights.item(), episode)


	def calculate_advantages_based_on_exp(self, V_values, V_values_old, rewards, dones, weights_prd, masks, episode):
		advantage = None
		masking_rewards = None
		mean_min_weight_value = -1
		if "shared" in self.experiment_type:
			rewards_ = torch.sum(rewards.unsqueeze(-2).repeat(1, self.num_agents, 1), dim=-1)
			advantage = self.calculate_advantages(V_values, V_values_old, rewards_, dones, masks)
		elif "prd_above_threshold_ascend" in self.experiment_type or "prd_above_threshold_decay" in self.experiment_type:
			masking_rewards = (weights_prd>self.select_above_threshold).int()
			rewards_ = torch.sum(rewards.unsqueeze(-2).repeat(1, self.num_agents, 1) * torch.transpose(masking_rewards,-1,-2), dim=-1)
			advantage = self.calculate_advantages(V_values, V_values_old, rewards_, dones, masks)
		elif "prd_above_threshold" in self.experiment_type:
			if episode > self.steps_to_take:
				masking_rewards = (weights_prd>self.select_above_threshold).int()
				rewards_ = torch.sum(rewards.unsqueeze(-2).repeat(1, self.num_agents, 1) * torch.transpose(masking_rewards,-1,-2), dim=-1)
			else:
				masking_rewards = torch.ones(weights_prd.shape).to(self.device)
				rewards_ = torch.sum(rewards.unsqueeze(-2).repeat(1, self.num_agents, 1), dim=-1)
			advantage = self.calculate_advantages(V_values, V_values_old, rewards_, dones, masks)
		elif "top" in self.experiment_type:
			if episode > self.steps_to_take:
				rewards_ = torch.sum(rewards.unsqueeze(-2).repeat(1, self.num_agents, 1), dim=-1)
				advantage = self.calculate_advantages(V_values, V_values_old, rewards_, dones, masks)
				masking_rewards = torch.ones(weights_prd.shape).to(self.device)
				min_weight_values, _ = torch.min(weights_prd, dim=-1)
				mean_min_weight_value = torch.mean(min_weight_values)
			else:
				values, indices = torch.topk(weights_prd,k=self.top_k,dim=-1)
				min_weight_values, _ = torch.min(values, dim=-1)
				mean_min_weight_value = torch.mean(min_weight_values)
				masking_rewards = torch.sum(F.one_hot(indices, num_classes=self.num_agents), dim=-2)
				rewards_ = torch.sum(masking_rewards * torch.transpose(masking_rewards,-1,-2), dim=-1)
				advantage = self.calculate_advantages(V_values, V_values_old, rewards_, dones, masks)
		elif "prd_soft_advantage" in self.experiment_type:
			if episode > self.steps_to_take:
				rewards_ = torch.sum(rewards.unsqueeze(-2).repeat(1, self.num_agents, 1) * torch.transpose(weights_prd, -1, -2), dim=-1)
			else:
				rewards_ = torch.sum(rewards.unsqueeze(-2).repeat(1, self.num_agents, 1), dim=-1)
			advantage = self.calculate_advantages(V_values, V_values_old, rewards_, dones, masks)
		
		if "scaled" in self.experiment_type and episode > self.steps_to_take and "top" in self.experiment_type:
			advantage *= self.num_agents
			if "top" in self.experiment_type:
				advantage /= self.top_k


		return advantage.detach(), masking_rewards, mean_min_weight_value


	def update_parameters(self):
		if self.select_above_threshold > self.threshold_min and "prd_above_threshold_decay" in self.experiment_type:
			self.select_above_threshold = self.select_above_threshold - self.threshold_delta

		if self.threshold_max > self.select_above_threshold and "prd_above_threshold_ascend" in self.experiment_type:
			self.select_above_threshold = self.select_above_threshold + self.threshold_delta

	def update(self, episode):
		# convert list to tensor
		old_states_critic = torch.FloatTensor(np.array(self.buffer.states_critic)).reshape(-1, self.num_agents, self.critic_observation_shape)
		old_rnn_hidden_state_v = torch.FloatTensor(np.array(self.buffer.rnn_hidden_state_v)).reshape(-1, self.num_agents, self.rnn_hidden_v)
		old_rnn_hidden_state_q = torch.FloatTensor(np.array(self.buffer.rnn_hidden_state_q)).reshape(-1, self.num_agents, self.rnn_hidden_q)
		old_states_actor = torch.FloatTensor(np.array(self.buffer.states_actor)).reshape(-1, self.num_agents, self.actor_observation_shape)
		old_rnn_hidden_state_actor = torch.FloatTensor(np.array(self.buffer.rnn_hidden_state_actor)).reshape(-1, self.num_agents, self.rnn_hidden_actor)
		old_actions = torch.FloatTensor(np.array(self.buffer.actions)).reshape(-1, self.num_agents)
		old_one_hot_actions = torch.FloatTensor(np.array(self.buffer.one_hot_actions)).reshape(-1, self.num_agents, self.num_actions)
		old_mask_actions = torch.FloatTensor(np.array(self.buffer.action_masks)).reshape(-1, self.num_agents, self.num_actions)
		old_logprobs = torch.FloatTensor(self.buffer.logprobs).reshape(-1, self.num_agents)
		rewards = torch.FloatTensor(np.array(self.buffer.rewards))
		dones = torch.FloatTensor(np.array(self.buffer.dones)).long()
		masks = torch.FloatTensor(np.array(self.buffer.masks)).long()

		with torch.no_grad():
			# OLD VALUES
			self.critic_network_q_old.rnn_hidden_state = old_rnn_hidden_state_q.to(self.device)
			Q_values_old, weights_prd_old, _, _ = self.critic_network_q_old(
												old_states_critic.to(self.device),
												old_one_hot_actions.to(self.device)
												)
			self.critic_network_v_old.rnn_hidden_state = old_rnn_hidden_state_v.to(self.device)
			Values_old, _, _, _ = self.critic_network_v_old(
												old_states_critic.to(self.device),
												old_one_hot_actions.to(self.device)
												)

		if "prd_above_threshold_ascend" in self.experiment_type or "prd_above_threshold_decay" in self.experiment_type:
			mask_rewards = (torch.mean(weights_prd_old, dim=1)>self.select_above_threshold).int()
			target_V_rewards = torch.sum(rewards.reshape(-1, self.num_agents).unsqueeze(-2).repeat(1, self.num_agents, 1) * torch.transpose(mask_rewards.detach().cpu(),-1,-2), dim=-1)
		elif "threshold" in self.experiment_type and episode > self.steps_to_take:
			mask_rewards = (torch.mean(weights_prd_old, dim=1)>self.select_above_threshold).int()
			target_V_rewards = torch.sum(rewards.reshape(-1, self.num_agents).unsqueeze(-2).repeat(1, self.num_agents, 1) * torch.transpose(mask_rewards.detach().cpu(),-1,-2), dim=-1)
		elif "top" in self.experiment_type and episode > self.steps_to_take:
			values, indices = torch.topk(torch.mean(weights_prd_old, dim=1), k=self.top_k,dim=-1)
			mask_rewards = torch.sum(F.one_hot(indices, num_classes=self.num_agents), dim=-2)
			target_V_rewards = torch.sum(rewards.reshape(-1, self.num_agents).unsqueeze(-2).repeat(1, self.num_agents, 1) * torch.transpose(mask_rewards.detach().cpu(),-1,-2), dim=-1)
		elif "prd_soft_advantage" in self.experiment_type and episode > self.steps_to_take:
			target_V_rewards = torch.sum(rewards.unsqueeze(-2).repeat(1, 1, self.num_agents, 1) * torch.transpose(torch.mean(weights_prd_old, dim=1).reshape(self.update_ppo_agent, self.max_time_steps, self.num_agents, self.num_agents), -1, -2), dim=-1)
		else:
			target_V_rewards = torch.sum(rewards.reshape(-1, self.num_agents).unsqueeze(-2).repeat(1, self.num_agents, 1), dim=-1)

		target_Q_values = self.build_td_lambda_targets(rewards.to(self.device), dones.to(self.device), masks.to(self.device), Q_values_old.reshape(self.update_ppo_agent, self.max_time_steps, self.num_agents)).reshape(-1, self.num_agents)
		target_V_values = self.build_td_lambda_targets(target_V_rewards.reshape(self.update_ppo_agent, self.max_time_steps, self.num_agents).to(self.device), dones.to(self.device), masks.to(self.device), Values_old.reshape(self.update_ppo_agent, self.max_time_steps, self.num_agents)).reshape(-1, self.num_agents)

		if self.norm_returns:
			target_Q_values = (target_Q_values - target_Q_values.mean()) / target_Q_values.std()
			target_V_values = (target_V_values - target_V_values.mean()) / target_V_values.std()
		
		rewards = rewards.reshape(-1, self.num_agents)
		dones = dones.reshape(-1, self.num_agents)
		masks = masks.reshape(-1, 1)

		q_value_loss_batch = 0
		v_value_loss_batch = 0
		policy_loss_batch = 0
		entropy_batch = 0
		weight_prd_batch = None
		weight_v_batch = None
		grad_norm_value_v_batch = 0
		grad_norm_value_q_batch = 0
		grad_norm_policy_batch = 0
		agent_groups_over_episode_batch = 0
		avg_agent_group_over_episode_batch = 0

		self.policy_network.rnn_hidden_state = old_rnn_hidden_state_actor.to(self.device)
		self.critic_network_v.rnn_hidden_state = old_rnn_hidden_state_v.to(self.device)
		self.critic_network_q.rnn_hidden_state = old_rnn_hidden_state_q.to(self.device)
		# torch.autograd.set_detect_anomaly(True)
		# Optimize policy for n epochs
		for _ in range(self.n_epochs):

			Q_value, weights_prd, score_q, rnn_hidden_state_q = self.critic_network_q(
				old_states_critic.to(self.device), 
				old_one_hot_actions.to(self.device)
				)
			Value, weight_v, score_v, rnn_hidden_state_v = self.critic_network_v(
				old_states_critic.to(self.device),  
				old_one_hot_actions.to(self.device)
				)

			print(torch.mean(weights_prd, dim=1))

			advantage, masking_rewards, mean_min_weight_value = self.calculate_advantages_based_on_exp(Value, Values_old, rewards.to(self.device), dones.to(self.device), torch.mean(weights_prd.detach(), dim=1), masks.to(self.device), episode)
			
			dists, rnn_hidden_state_actor = self.policy_network(old_states_actor.to(self.device), old_mask_actions.to(self.device))
			probs = Categorical(dists.squeeze(0))
			logprobs = probs.log_prob(old_actions.to(self.device))

			if "threshold" in self.experiment_type or "top" in self.experiment_type:
				agent_groups_over_episode = torch.sum(torch.sum(masking_rewards.float(), dim=-2),dim=0)/masking_rewards.shape[0]
				avg_agent_group_over_episode = torch.mean(agent_groups_over_episode)
				agent_groups_over_episode_batch += agent_groups_over_episode
				avg_agent_group_over_episode_batch += avg_agent_group_over_episode
				
			
			critic_v_loss_1 = F.mse_loss(Value*masks.to(self.device), target_V_values*masks.to(self.device), reduction="sum") / masks.sum()
			critic_v_loss_2 = F.mse_loss(torch.clamp(Value, Values_old.to(self.device)-self.value_clip, Values_old.to(self.device)+self.value_clip)*masks.to(self.device), target_V_values*masks.to(self.device), reduction="sum") / masks.sum()

			critic_q_loss_1 = F.mse_loss(Q_value*masks.to(self.device), target_Q_values*masks.to(self.device), reduction="sum") / masks.sum()
			critic_q_loss_2 = F.mse_loss(torch.clamp(Q_value, Q_values_old.to(self.device)-self.value_clip, Q_values_old.to(self.device)+self.value_clip)*masks.to(self.device), target_Q_values*masks.to(self.device), reduction="sum") / masks.sum()

			# Finding the ratio (pi_theta / pi_theta__old)
			ratios = torch.exp(logprobs - old_logprobs.to(self.device))
			# Finding Surrogate Loss
			surr1 = ratios * advantage*masks.unsqueeze(-1).to(self.device)
			surr2 = torch.clamp(ratios, 1-self.policy_clip, 1+self.policy_clip) * advantage * masks.unsqueeze(-1).to(self.device)

			# final loss of clipped objective PPO
			entropy = -torch.mean(torch.sum(dists*masks.unsqueeze(-1).to(self.device) * torch.log(torch.clamp(dists*masks.unsqueeze(-1).to(self.device), 1e-10,1.0)), dim=2))
			policy_loss = (-torch.min(surr1, surr2).mean() - self.entropy_pen*entropy)
			
			entropy_weights = 0
			entropy_weights_v = 0
			for i in range(self.num_heads):
				entropy_weights += -torch.mean(torch.sum(weights_prd[:, i] * torch.log(torch.clamp(weights_prd[:, i], 1e-10,1.0)), dim=2))
				entropy_weights_v += -torch.mean(torch.sum(weight_v[:, i] * torch.log(torch.clamp(weight_v[:, i], 1e-10,1.0)), dim=2))
				
			critic_q_loss = torch.max(critic_q_loss_1, critic_q_loss_2) + self.critic_score_regularizer*(score_q**2).sum(dim=-1).mean() + self.critic_weight_entropy_pen*entropy_weights
			critic_v_loss = torch.max(critic_v_loss_1, critic_v_loss_2) + self.critic_score_regularizer*(score_v**2).sum(dim=-1).mean() + self.critic_weight_entropy_pen*entropy_weights_v

			
			self.q_critic_optimizer.zero_grad()
			critic_q_loss.backward()
			if self.enable_grad_clip_critic:
				grad_norm_value_q = torch.nn.utils.clip_grad_norm_(self.critic_network_q.parameters(), self.grad_clip_critic)
			else:
				grad_norm_value_q = torch.tensor([-1.0])
			self.q_critic_optimizer.step()

			self.critic_network_q.rnn_hidden_state = rnn_hidden_state_q.detach()
			
			self.v_critic_optimizer.zero_grad()
			critic_v_loss.backward()
			if self.enable_grad_clip_critic:
				grad_norm_value_v = torch.nn.utils.clip_grad_norm_(self.critic_network_v.parameters(), self.grad_clip_critic)
			else:
				grad_norm_value_v = torch.tensor([-1.0])
			self.v_critic_optimizer.step()

			self.critic_network_v.rnn_hidden_state = rnn_hidden_state_v.detach()
			

			self.policy_optimizer.zero_grad()
			policy_loss.backward()
			if self.enable_grad_clip_actor:
				grad_norm_policy = torch.nn.utils.clip_grad_norm_(self.policy_network.parameters(), self.grad_clip_actor)
			else:
				grad_norm_policy = torch.tensor([-1.0])
			self.policy_optimizer.step()

			self.policy_network.rnn_hidden_state = rnn_hidden_state_actor.detach()

			q_value_loss_batch += critic_q_loss.item()
			v_value_loss_batch += critic_v_loss.item()
			policy_loss_batch += policy_loss.item()
			entropy_batch += entropy.item()
			grad_norm_value_v_batch += grad_norm_value_v
			grad_norm_value_q_batch += grad_norm_value_q
			grad_norm_policy_batch += grad_norm_policy
			if weight_prd_batch is None:
				weight_prd_batch = weights_prd.detach().cpu()
			else:
				weight_prd_batch += weights_prd.detach().cpu()
			if weight_v_batch is None:
				weight_v_batch = weight_v.detach().cpu()
			else:
				weight_v_batch += weight_v.detach().cpu()
			


		if episode % self.network_update_interval == 0:
			# Copy new weights into old policy
			self.policy_network_old.load_state_dict(self.policy_network.state_dict())

			# Copy new weights into old critic
			self.critic_network_q_old.load_state_dict(self.critic_network_q.state_dict())
			self.critic_network_v_old.load_state_dict(self.critic_network_v.state_dict())

		# self.scheduler.step()
		# print("learning rate of policy", self.scheduler.get_lr())

		# clear buffer
		self.buffer.clear()
		self.policy_network.rnn_hidden_state = None
		self.policy_network_old.rnn_hidden_state = None
		
		self.critic_network_q_old.rnn_hidden_state = None
		self.critic_network_q.rnn_hidden_state = None

		self.critic_network_v_old.rnn_hidden_state = None
		self.critic_network_v.rnn_hidden_state = None

		q_value_loss_batch /= self.n_epochs
		v_value_loss_batch /= self.n_epochs
		policy_loss_batch /= self.n_epochs
		entropy_batch /= self.n_epochs
		grad_norm_value_v_batch /= self.n_epochs
		grad_norm_value_q_batch /= self.n_epochs
		grad_norm_policy_batch /= self.n_epochs
		weight_prd_batch /= self.n_epochs
		weight_v_batch /= self.n_epochs
		agent_groups_over_episode_batch /= self.n_epochs
		avg_agent_group_over_episode_batch /= self.n_epochs


		if "prd" in self.experiment_type:
			if self.update_learning_rate_with_prd:
				for g in self.policy_optimizer.param_groups:
					g['lr'] = self.policy_lr * self.num_agents/avg_agent_group_over_episode_batch

		self.update_parameters()


		self.plotting_dict = {
		"q_value_loss": q_value_loss_batch,
		"v_value_loss": v_value_loss_batch,
		"policy_loss": policy_loss_batch,
		"entropy": entropy_batch,
		"grad_norm_value_v":grad_norm_value_v_batch,
		"grad_norm_value_q": grad_norm_value_q_batch,
		"grad_norm_policy": grad_norm_policy_batch,
		"weights_prd": weight_prd_batch,
		"weights_v": weight_v_batch,
		}

		if "threshold" in self.experiment_type:
			self.plotting_dict["agent_groups_over_episode"] = agent_groups_over_episode_batch
			self.plotting_dict["avg_agent_group_over_episode"] = avg_agent_group_over_episode_batch
		if "prd_top" in self.experiment_type:
			self.plotting_dict["mean_min_weight_value"] = mean_min_weight_value

		if self.comet_ml is not None:
			self.plot(episode)

		del q_value_loss_batch, v_value_loss_batch, policy_loss_batch, entropy_batch, grad_norm_value_v_batch, grad_norm_value_q_batch, grad_norm_policy_batch, weight_prd_batch, agent_groups_over_episode_batch, avg_agent_group_over_episode_batch
		torch.cuda.empty_cache()
