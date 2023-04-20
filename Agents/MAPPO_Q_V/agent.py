import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from model import MLP_Policy, Q_V_network, Q_network, V_network
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
		self.num_actions = 5

		# Training setup
		self.test_num = dictionary["test_num"]
		self.gif = dictionary["gif"]
		self.experiment_type = dictionary["experiment_type"]
		self.false_positive_rate = []
		self.n_epochs = dictionary["n_epochs"]
		self.scheduler_need = dictionary["scheduler_need"]
		if dictionary["device"] == "gpu":
			self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		else:
			self.device = "cpu"

		self.update_ppo_agent = dictionary["update_ppo_agent"]
		self.max_time_steps = dictionary["max_time_steps"]
		
		# Critic Setup
		self.critic_observation_shape = dictionary["global_observation"]
		self.value_lr = dictionary["value_lr"]
		self.q_value_lr = dictionary["q_value_lr"]
		self.critic_weight_entropy_pen = dictionary["critic_weight_entropy_pen"]
		self.critic_score_regularizer = dictionary["critic_score_regularizer"]
		self.lambda_ = dictionary["lambda"] # TD lambda
		self.value_clip = dictionary["value_clip"]
		self.num_heads = dictionary["num_heads"]
		self.enable_hard_attention = dictionary["enable_hard_attention"]
		self.grad_clip_critic = dictionary["grad_clip_critic"]


		# Actor Setup
		self.actor_observation_shape = dictionary["local_observation"]
		self.policy_lr = dictionary["policy_lr"]
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
		self.critic_network_q = Q_network(obs_input_dim=self.critic_observation_shape, num_heads=self.num_heads, num_agents=self.num_agents, num_actions=self.num_actions, device=self.device, enable_hard_attention=self.enable_hard_attention).to(self.device)
		self.critic_network_q_old = Q_network(obs_input_dim=self.critic_observation_shape, num_heads=self.num_heads, num_agents=self.num_agents, num_actions=self.num_actions, device=self.device, enable_hard_attention=self.enable_hard_attention).to(self.device)
		# Copy network params
		self.critic_network_q_old.load_state_dict(self.critic_network_q.state_dict())
		# Disable updates for old network
		for param in self.critic_network_q_old.parameters():
			param.requires_grad_(False)

		self.history_states_critic_q = None

		self.critic_network_v = V_network(obs_input_dim=self.critic_observation_shape, num_heads=self.num_heads, num_agents=self.num_agents, num_actions=self.num_actions, device=self.device, enable_hard_attention=self.enable_hard_attention).to(self.device)
		self.critic_network_v_old = V_network(obs_input_dim=self.critic_observation_shape, num_heads=self.num_heads, num_agents=self.num_agents, num_actions=self.num_actions, device=self.device, enable_hard_attention=self.enable_hard_attention).to(self.device)
		# Copy network params
		self.critic_network_v_old.load_state_dict(self.critic_network_v.state_dict())
		# Disable updates for old network
		for param in self.critic_network_v_old.parameters():
			param.requires_grad_(False)

		self.history_states_critic_v = None
		
		
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
			obs_shape=self.critic_observation_shape, 
			num_actions=self.num_actions
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

		# params_to_update = []
		# for name, param in self.critic_network.named_parameters():
		# 	if param.requires_grad == True:
		# 		if "v_value_layer" in name:
		# 			continue
		# 		params_to_update.append(param)
				# print("\t",name)

		self.q_critic_optimizer = optim.Adam(self.critic_network_q.parameters(), lr=self.q_value_lr)
		self.v_critic_optimizer = optim.Adam(self.critic_network_v.parameters(), lr=self.value_lr)
		self.policy_optimizer = optim.Adam(self.policy_network.parameters(),lr=self.policy_lr)

		if self.scheduler_need:
			self.scheduler_policy = optim.lr_scheduler.MultiStepLR(self.policy_optimizer, milestones=[1000, 20000], gamma=0.1)
			self.scheduler_q_critic = optim.lr_scheduler.MultiStepLR(self.q_critic_optimizer, milestones=[1000, 20000], gamma=0.1)
			self.scheduler_v_critic = optim.lr_scheduler.MultiStepLR(self.v_critic_optimizer, milestones=[1000, 20000], gamma=0.1)

		self.comet_ml = None
		if dictionary["save_comet_ml_plot"]:
			self.comet_ml = comet_ml


	def get_action(self, state_policy, greedy=False):
		with torch.no_grad():
			state_policy = torch.FloatTensor(state_policy).to(self.device)
			dists = self.policy_network_old(state_policy)
			if greedy:
				actions = [dist.argmax().detach().cpu().item() for dist in dists]
			else:
				actions = [Categorical(dist).sample().detach().cpu().item() for dist in dists]

			probs = Categorical(dists)
			action_logprob = probs.log_prob(torch.FloatTensor(actions).to(self.device))

			return actions, action_logprob.detach().cpu().numpy()


	def calculate_advantages(self, values, values_old, rewards, dones):
		advantages = []
		next_value = 0
		advantage = 0
		masks = 1 - dones
		for t in reversed(range(0, len(rewards))):
			td_error = rewards[t] + (self.gamma * next_value * masks[t]) - values.data[t]
			next_value = values_old.data[t]
			
			advantage = td_error + (self.gamma * self.gae_lambda * advantage * masks[t])
			advantages.insert(0, advantage)

		advantages = torch.stack(advantages)
		
		if self.norm_adv:
			advantages = (advantages - advantages.mean()) / advantages.std()
		
		return advantages


	def build_td_lambda_targets(self, rewards, terminated, mask, target_qs):
		# Assumes  <target_qs > in B*T*A and <reward >, <terminated >  in B*T*A, <mask > in (at least) B*T-1*1
		# Initialise  last  lambda -return  for  not  terminated  episodes
		ret = target_qs.new_zeros(*target_qs.shape)
		ret[:, -1] = target_qs[:, -1] * (1 - torch.sum(terminated, dim=1))
		# Backwards  recursive  update  of the "forward  view"
		for t in range(ret.shape[1] - 2, -1,  -1):
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

		if "prd_top" in self.experiment_type:
			self.comet_ml.log_metric('Mean_Smallest_Weight', self.plotting_dict["mean_min_weight_value"].item(), episode)


		# ENTROPY OF Q WEIGHTS
		for i in range(self.num_heads):
			entropy_weights = -torch.mean(torch.sum(self.plotting_dict["weights_prd"][:,i]* torch.log(torch.clamp(self.plotting_dict["weights_prd"][:,i], 1e-10,1.0)), dim=2))
			self.comet_ml.log_metric('Q_Weight_Entropy_Head_'+str(i+1), entropy_weights.item(), episode)

		# ENTROPY OF V WEIGHTS
		for i in range(self.num_heads):
			entropy_weights = -torch.mean(torch.sum(self.plotting_dict["weights_v"][:,i]* torch.log(torch.clamp(self.plotting_dict["weights_v"][:,i], 1e-10,1.0)), dim=2))
			self.comet_ml.log_metric('V_Weight_Entropy_Head_'+str(i+1), entropy_weights.item(), episode)


	def calculate_advantages_based_on_exp(self, V_values, V_values_old, rewards, dones, weights_prd, episode):
		advantage = None
		masking_rewards = None
		mean_min_weight_value = -1
		if "shared" in self.experiment_type:
			rewards_ = torch.sum(rewards.unsqueeze(-2).repeat(1, self.num_agents, 1), dim=-1)
			advantage = torch.sum(self.calculate_advantages(V_values, V_values_old, rewards_, dones),dim=-2)
		elif "prd_above_threshold_ascend" in self.experiment_type or "prd_above_threshold_decay" in self.experiment_type:
			masking_rewards = (weights_prd>self.select_above_threshold).int()
			rewards_ = torch.sum(rewards.unsqueeze(-2).repeat(1, self.num_agents, 1) * torch.transpose(masking_rewards,-1,-2), dim=-1)
			advantage = self.calculate_advantages(V_values, V_values_old, rewards_, dones)
		elif "prd_above_threshold" in self.experiment_type:
			if episode > self.steps_to_take:
				masking_rewards = (weights_prd>self.select_above_threshold).int()
				rewards_ = torch.sum(rewards.unsqueeze(-2).repeat(1, self.num_agents, 1) * torch.transpose(masking_rewards,-1,-2), dim=-1)
			else:
				masking_rewards = torch.ones(weights_prd.shape).to(self.device)
				rewards_ = torch.sum(rewards.unsqueeze(-2).repeat(1, self.num_agents, 1), dim=-1)
			advantage = self.calculate_advantages(V_values, V_values_old, rewards_, dones)
		elif "top" in self.experiment_type:
			if episode > self.steps_to_take:
				rewards_ = torch.sum(rewards.unsqueeze(-2).repeat(1, self.num_agents, 1), dim=-1)
				advantage = torch.sum(self.calculate_advantages(V_values, V_values_old, rewards_, dones),dim=-2)
				masking_rewards = torch.ones(weights_prd.shape).to(self.device)
				min_weight_values, _ = torch.min(weights_prd, dim=-1)
				mean_min_weight_value = torch.mean(min_weight_values)
			else:
				values, indices = torch.topk(weights_prd,k=self.top_k,dim=-1)
				min_weight_values, _ = torch.min(values, dim=-1)
				mean_min_weight_value = torch.mean(min_weight_values)
				masking_rewards = torch.sum(F.one_hot(indices, num_classes=self.num_agents), dim=-2)
				rewards_ = torch.sum(masking_rewards * torch.transpose(masking_rewards,-1,-2), dim=-1)
				advantage = self.calculate_advantages(V_values, V_values_old, rewards_, dones)
		elif "greedy" in self.experiment_type:
			advantage = self.calculate_advantages(V_values, V_values_old, rewards, dones)

		if "scaled" in self.experiment_type and episode > self.steps_to_take and "top" in self.experiment_type:
			advantage = advantage*(self.num_agents/self.top_k)

		return advantage.detach(), masking_rewards, mean_min_weight_value


	# def calculate_advantages_Q_V(self, Q, V, weights_prd, dones, episode):
	# 	'''
	# 	Q : B x N x 1
	# 	V : B x N x 1
	# 	weights_prd: B x N x N x 1
	# 	dones: B x N x 1
	# 	episode : int
	# 	'''
	# 	advantage = None
	# 	masking_advantage = None
	# 	mean_min_weight_value = -1
	# 	dones = dones.unsqueeze(-1)

	# 	if "shared" in self.experiment_type:
	# 		advantage = torch.sum(Q.unsqueeze(1).repeat(1, self.num_agents, 1) * (1-dones), dim=-1) - V # B x N x 1

	# 	elif "prd_above_threshold" in self.experiment_type:
	# 		# No masking until warm-up period ends
	# 		if episode < self.steps_to_take:
	# 			masking_advantage = torch.ones(weights_prd.shape).to(self.device)
	# 			advantage = torch.sum(Q.unsqueeze(1).repeat(1, self.num_agents, 1) * torch.transpose(masking_advantage,-1,-2) * (1-dones), dim=-1) - V # B x N x 1
	# 		else:
	# 			masking_advantage = (weights_prd > self.select_above_threshold).int()
	# 			advantage = torch.sum(Q.unsqueeze(1).repeat(1, self.num_agents, 1) * torch.transpose(masking_advantage,-1,-2) * (1-dones), dim=-1) - V # B x N x 1

	# 	elif "top" in self.experiment_type:
	# 		# No masking until warm-up period ends
	# 		if episode < self.steps_to_take:
	# 			masking_advantage = torch.ones(weights_prd.shape).to(self.device)
	# 			advantage = torch.sum(Q.unsqueeze(1).repeat(1, self.num_agents, 1) * torch.transpose(masking_advantage,-1,-2) * (1-dones), dim=-1) - V # B x N x 1
	# 			min_weight_values, _ = torch.min(weights_prd, dim=-1)
	# 			mean_min_weight_value = torch.mean(min_weight_values)
	# 		else:
	# 			values, indices = torch.topk(weights_prd,k=self.top_k,dim=-1)
	# 			min_weight_values, _ = torch.min(values, dim=-1)
	# 			mean_min_weight_value = torch.mean(min_weight_values)
	# 			masking_advantage = torch.sum(F.one_hot(indices, num_classes=self.num_agents), dim=-2)
	# 			advantage = torch.sum(Q.unsqueeze(1).repeat(1, self.num_agents, 1) * torch.transpose(masking_advantage,-1,-2) * (1-dones), dim=-1) - V # B x N x 1
		
	# 	if "scaled" in self.experiment_type and episode > self.steps_to_take and "top" in self.experiment_type:
	# 		advantage = advantage*(self.num_agents/self.top_k)

	# 	return advantage.detach(), masking_advantage, mean_min_weight_value

	def update_parameters(self):
		if self.select_above_threshold > self.threshold_min and "prd_above_threshold_decay" in self.experiment_type:
			self.select_above_threshold = self.select_above_threshold - self.threshold_delta

		if self.threshold_max > self.select_above_threshold and "prd_above_threshold_ascend" in self.experiment_type:
			self.select_above_threshold = self.select_above_threshold + self.threshold_delta



	def update(self, episode):
		# convert list to tensor
		old_states = torch.FloatTensor(np.array(self.buffer.states)).reshape(-1, self.num_agents, self.critic_observation_shape)
		old_actions = torch.FloatTensor(np.array(self.buffer.actions)).reshape(-1, self.num_agents)
		old_one_hot_actions = torch.FloatTensor(np.array(self.buffer.one_hot_actions)).reshape(-1, self.num_agents, self.num_actions)
		old_logprobs = torch.FloatTensor(self.buffer.logprobs).reshape(-1, self.num_agents)
		rewards = torch.FloatTensor(np.array(self.buffer.rewards))
		dones = torch.FloatTensor(np.array(self.buffer.dones)).long()
		masks = torch.FloatTensor(np.array(self.buffer.masks)).long()

		# if self.history_states_critic_q is None:
		self.history_states_critic_q = torch.zeros(old_states.shape[0], self.num_agents, 64)

		# if self.history_states_critic_v is None:
		self.history_states_critic_v = torch.zeros(old_states.shape[0], self.num_agents, 64)

		with torch.no_grad():
			# OLD VALUES
			Q_values_old, self.history_states_critic_q, weights_prd_old, _ = self.critic_network_q_old(
																old_states.to(self.device),
																self.history_states_critic_q.to(self.device),
																old_one_hot_actions.to(self.device)
																)

			Values_old, self.history_states_critic_v, _, _ = self.critic_network_v_old(
													old_states.reshape(-1, self.num_agents, self.critic_observation_shape).to(self.device),
													self.history_states_critic_v.to(self.device),
													old_one_hot_actions.reshape(-1, self.num_agents, self.num_actions).to(self.device)
													)

		if "threshold" in self.experiment_type or "top" in self.experiment_type:
			mask_rewards = (torch.mean(weights_prd_old, dim=1)>self.select_above_threshold).int()
			target_V_rewards = torch.sum(rewards.reshape(-1, self.num_agents).unsqueeze(-2).repeat(1, self.num_agents, 1) * torch.transpose(mask_rewards.detach().cpu(),-1,-2), dim=-1)
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

		# torch.autograd.set_detect_anomaly(True)
		# Optimize policy for n epochs
		for _ in range(self.n_epochs):

			Q_value, new_history_states_critic_q, weights_prd, score_q = self.critic_network_q(
				old_states.to(self.device), 
				self.history_states_critic_q.to(self.device), 
				old_one_hot_actions.to(self.device)
				)
			Value, new_history_states_critic_v, weight_v, score_v = self.critic_network_v(
				old_states.to(self.device), 
				self.history_states_critic_v.to(self.device), 
				old_one_hot_actions.to(self.device)
				)

			advantage, masking_rewards, mean_min_weight_value = self.calculate_advantages_based_on_exp(Value, Values_old, rewards.to(self.device), dones.to(self.device), torch.mean(weights_prd.detach(), dim=1), episode)

			dists = self.policy_network(old_states.to(self.device))
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

			print(critic_v_loss_1, critic_v_loss_2, masks.sum())
			print(critic_q_loss_1, critic_q_loss_2, masks.sum())

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

			critic_q_loss = torch.max(critic_q_loss_1, critic_q_loss_2) + self.critic_score_regularizer*(score_q**2).mean() + self.critic_weight_entropy_pen*entropy_weights
			critic_v_loss = torch.max(critic_v_loss_1, critic_v_loss_2) + self.critic_score_regularizer*(score_v**2).mean() + self.critic_weight_entropy_pen*entropy_weights_v

			# take gradient step
			# self.critic_optimizer.zero_grad()
			# (critic_q_loss + critic_v_loss).backward(retain_graph=True)
			# grad_norm_value = torch.nn.utils.clip_grad_norm_(self.critic_network.parameters(), self.grad_clip_critic)
			# self.critic_optimizer.step()

			
			self.q_critic_optimizer.zero_grad()
			critic_q_loss.backward()
			grad_norm_value_q = torch.nn.utils.clip_grad_norm_(self.critic_network_q.parameters(), self.grad_clip_critic)
			self.q_critic_optimizer.step()
			
			self.v_critic_optimizer.zero_grad()
			critic_v_loss.backward()
			grad_norm_value_v = torch.nn.utils.clip_grad_norm_(self.critic_network_v.parameters(), self.grad_clip_critic)
			self.v_critic_optimizer.step()
			

			self.policy_optimizer.zero_grad()
			policy_loss.backward()
			grad_norm_policy = torch.nn.utils.clip_grad_norm_(self.policy_network.parameters(), self.grad_clip_actor)
			self.policy_optimizer.step()

			self.history_states_critic_q = new_history_states_critic_q.detach().cpu()
			self.history_states_critic_v = new_history_states_critic_v.detach().cpu()

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

		# return history_states_critic[-1]
