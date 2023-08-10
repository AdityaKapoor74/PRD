import numpy as np
import time
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
		self.critic_observation_shape = dictionary["global_observation"]
		self.value_lr = dictionary["value_lr"]
		self.value_weight_decay = dictionary["value_weight_decay"]
		self.critic_weight_entropy_pen = dictionary["critic_weight_entropy_pen"]
		self.critic_score_regularizer = dictionary["critic_score_regularizer"]
		self.lambda_ = dictionary["lambda"] # TD lambda
		self.value_clip = dictionary["value_clip"]
		self.num_heads = dictionary["num_heads"]
		self.enable_grad_clip_critic = dictionary["enable_grad_clip_critic"]
		self.grad_clip_critic = dictionary["grad_clip_critic"]


		# Actor Setup
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
		
		self.select_above_threshold = dictionary["select_above_threshold"]
		if "prd_above_threshold_decay" in self.experiment_type:
			self.threshold_delta = (self.select_above_threshold - self.threshold_min)/self.steps_to_take
		elif "prd_above_threshold_ascend" in self.experiment_type:
			self.threshold_delta = (self.threshold_max - self.select_above_threshold)/self.steps_to_take

		print("EXPERIMENT TYPE", self.experiment_type)

		self.critic_network = TransformerCritic(
			obs_input_dim=self.critic_observation_shape,
			obs_act_input_dim=self.critic_observation_shape+self.num_actions, 
			num_agents=self.num_agents, 
			num_actions=self.num_actions, 
			device=self.device).to(self.device)
		self.target_critic_network = TransformerCritic(
			obs_input_dim=self.critic_observation_shape,
			obs_act_input_dim=self.critic_observation_shape+self.num_actions, 
			num_agents=self.num_agents, 
			num_actions=self.num_actions, 
			device=self.device).to(self.device)

		self.target_critic_network.load_state_dict(self.critic_network.state_dict())

		# Disable updates for old network
		for param in self.target_critic_network.parameters():
			param.requires_grad_(False)

		self.policy_network = MLP_Policy(obs_input_dim=self.actor_observation_shape, 
			num_agents=self.num_agents, 
			num_actions=self.num_actions, 
			device=self.device).to(self.device)

		self.target_policy_network = MLP_Policy(obs_input_dim=self.actor_observation_shape, 
			num_agents=self.num_agents, 
			num_actions=self.num_actions, 
			device=self.device).to(self.device)

		self.target_critic_network.load_state_dict(self.critic_network.state_dict())

		# Disable updates for old network
		for param in self.target_policy_network.parameters():
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

		
		self.critic_optimizer = optim.Adam(self.critic_network.parameters(),lr=self.value_lr, weight_decay=self.value_weight_decay)
		self.policy_optimizer = optim.Adam(self.policy_network.parameters(),lr=self.policy_lr, weight_decay=self.policy_weight_decay)

		if self.scheduler_need:
			self.scheduler_policy = optim.lr_scheduler.MultiStepLR(self.policy_optimizer, milestones=[1000, 20000], gamma=0.1)
			self.scheduler_value = optim.lr_scheduler.MultiStepLR(self.critic_optimizer, milestones=[1000, 20000], gamma=0.1)


		self.comet_ml = None
		if dictionary["save_comet_ml_plot"]:
			self.comet_ml = comet_ml


	def get_actions(self, state, mask_actions):
		with torch.no_grad():
			state = torch.FloatTensor(state).to(self.device)
			mask_actions = torch.FloatTensor(mask_actions).to(self.device)
			dists, rnn_hidden_state = self.target_policy_network(state, mask_actions)
			actions = [Categorical(dist).sample().detach().cpu().item() for dist in dists]

			return actions, rnn_hidden_state.cpu().numpy()


	def calculate_advantages(self, target_values, values, rewards, dones, masks_):
		
		advantages = []
		next_value = 0
		advantage = 0
		rewards = rewards.unsqueeze(-1)
		dones = dones.unsqueeze(-1)
		masks_ = masks_.unsqueeze(-1)
		masks = 1 - dones
		for t in reversed(range(0, len(rewards))):
			td_error = rewards[t] + (self.gamma * next_value * masks[t]) - values.data[t]
			next_value = target_values.data[t]
			
			advantage = td_error + (self.gamma * self.gae_lambda * advantage * masks[t]) * masks_[t]
			advantages.insert(0, advantage)

		advantages = torch.stack(advantages)
		
		if self.norm_adv:
			advantages = (advantages - advantages.mean()) / advantages.std()
		
		return advantages


	def calculate_returns(self,rewards, discount_factor):
		returns = []
		R = 0
		
		for r in reversed(rewards):
			R = r + R * discount_factor
			returns.insert(0, R)
		
		returns_tensor = torch.stack(returns).to(self.device)
		
		if self.norm_rew:
			
			returns_tensor = (returns_tensor - returns_tensor.mean()) / returns_tensor.std()
			
		return returns_tensor

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
	
		self.comet_ml.log_metric('V_Value_Loss',self.plotting_dict["value_loss"],episode)
		self.comet_ml.log_metric('Grad_Norm_V_Value',self.plotting_dict["grad_norm_value"],episode)
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


		entropy_weights = -torch.mean(torch.sum(self.plotting_dict["weights_value"]* torch.log(torch.clamp(self.plotting_dict["weights_value"], 1e-10,1.0)), dim=2))
		self.comet_ml.log_metric('Critic_Weight_Entropy', entropy_weights.item(), episode)


	def calculate_advantages_based_on_exp(self, target_values, V_values, rewards, dones, weights_prd, masks, episode):
		# summing across each agent j to get the advantage
		# so we sum across the second last dimension which does A[t,j] = sum(V[t,i,j] - target_values[t,i])
		advantage = None
		masking_advantage = None
		if "shared" in self.experiment_type:
			advantage = torch.sum(self.calculate_advantages(target_values, V_values, rewards, dones, masks),dim=-2)
		elif "prd_soft_adv" in self.experiment_type:
			if episode < self.steps_to_take:
				advantage = torch.sum(self.calculate_advantages(target_values, V_values, rewards, dones, masks),dim=-2)
			else:
				advantage = torch.sum(self.calculate_advantages(target_values, V_values, rewards, dones, masks) * torch.transpose(weights_prd,-1,-2) ,dim=-2)
		elif "prd_averaged" in self.experiment_type:
			avg_weights = torch.mean(weights_prd,dim=0)
			advantage = torch.sum(self.calculate_advantages(target_values, V_values, rewards, dones, masks) * torch.transpose(avg_weights,-1,-2) ,dim=-2)
		elif "prd_avg_top" in self.experiment_type:
			avg_weights = torch.mean(weights_prd,dim=0)
			values, indices = torch.topk(avg_weights,k=self.top_k,dim=-1)
			masking_advantage = torch.sum(F.one_hot(indices, num_classes=self.num_agents), dim=-2)
			advantage = torch.sum(self.calculate_advantages(target_values, V_values, rewards, dones, masks) * torch.transpose(masking_advantage,-1,-2),dim=-2)
		elif "prd_avg_above_threshold" in self.experiment_type:
			avg_weights = torch.mean(weights_prd,dim=0)
			masking_advantage = (avg_weights>self.select_above_threshold).int()
			advantage = torch.sum(self.calculate_advantages(target_values, V_values, rewards, dones, masks) * torch.transpose(masking_advantage,-1,-2),dim=-2)
		elif "prd_above_threshold" in self.experiment_type:
			masking_advantage = (weights_prd>self.select_above_threshold).int()
			advantage = torch.sum(self.calculate_advantages(target_values, V_values, rewards, dones, masks) * torch.transpose(masking_advantage,-1,-2),dim=-2)
		elif "top" in self.experiment_type:
			if episode < self.steps_to_take:
				advantage = torch.sum(self.calculate_advantages(target_values, V_values, rewards, dones, masks),dim=-2)
				min_weight_values, _ = torch.min(weights_prd, dim=-1)
				mean_min_weight_value = torch.mean(min_weight_values)
			else:
				values, indices = torch.topk(weights_prd,k=self.top_k,dim=-1)
				min_weight_values, _ = torch.min(values, dim=-1)
				mean_min_weight_value = torch.mean(min_weight_values)
				masking_advantage = torch.sum(F.one_hot(indices, num_classes=self.num_agents), dim=-2)
				advantage = torch.sum(self.calculate_advantages(target_values, V_values, rewards, dones, masks) * torch.transpose(masking_advantage,-1,-2),dim=-2)
		elif "greedy" in self.experiment_type:
			advantage = torch.sum(self.calculate_advantages(target_values, V_values, rewards, dones, masks) * self.greedy_policy ,dim=-2)
		elif "relevant_set" in self.experiment_type:
			advantage = torch.sum(self.calculate_advantages(target_values, V_values, rewards, dones, masks) * self.relevant_set ,dim=-2)

		if "scaled" in self.experiment_type and episode > self.steps_to_take:
			if "prd_soft_adv" in self.experiment_type:
				advantage = advantage*self.num_agents
			elif "top" in self.experiment_type:
				advantage = advantage*(self.num_agents/self.top_k)

		return advantage, masking_advantage


	def update_parameters(self):

		if self.select_above_threshold > self.threshold_min and "prd_above_threshold_decay" in self.experiment_type:
			self.select_above_threshold = self.select_above_threshold - self.threshold_delta

		if self.threshold_max >= self.select_above_threshold and "prd_above_threshold_ascend" in self.experiment_type:
			self.select_above_threshold = self.select_above_threshold + self.threshold_delta


	def update(self, episode):

		old_states_critic = torch.FloatTensor(np.array(self.buffer.states_critic)).reshape(-1, self.num_agents, self.critic_observation_shape)
		old_states_actor = torch.FloatTensor(np.array(self.buffer.states_actor)).reshape(-1, self.num_agents, self.actor_observation_shape)
		old_rnn_hidden_state_actor = torch.FloatTensor(np.array(self.buffer.rnn_hidden_state_actor)).reshape(-1, self.num_agents, self.rnn_hidden_actor)
		old_actions = torch.FloatTensor(np.array(self.buffer.actions)).reshape(-1, self.num_agents)
		old_one_hot_actions = torch.FloatTensor(np.array(self.buffer.one_hot_actions)).reshape(-1, self.num_agents, self.num_actions)
		old_mask_actions = torch.FloatTensor(np.array(self.buffer.mask_actions)).reshape(-1, self.num_agents, self.num_actions)
		rewards = torch.FloatTensor(np.array(self.buffer.rewards))
		dones = torch.FloatTensor(np.array(self.buffer.dones)).long()
		masks = torch.FloatTensor(np.array(self.buffer.masks)).long()

		with torch.no_grad():
			self.target_policy_network.rnn_hidden_state = old_rnn_hidden_state_actor.to(self.device)
			old_dists, _ = self.target_policy_network(old_states_actor.to(self.device), old_mask_actions.to(self.device))
			old_probs = Categorical(old_dists.squeeze(0))
			old_logprobs = old_probs.log_prob(old_actions.to(self.device))
			V_values_old, _ = self.target_critic_network(old_states_critic.to(self.device), old_dists, old_one_hot_actions.to(self.device))
			V_values_old = V_values_old.reshape(-1, self.num_agents, self.num_agents)

		target_V_values = self.build_td_lambda_targets(rewards.unsqueeze(-1).to(self.device), dones.unsqueeze(-1).to(self.device), masks.unsqueeze(-1).to(self.device), V_values_old.reshape(self.update_ppo_agent, -1, self.num_agents, self.num_agents)).reshape(-1, self.num_agents, self.num_agents)

		value_loss_batch = 0.0
		policy_loss_batch = 0.0
		entropy_batch = 0.0
		grad_norm_value_batch = 0.0
		grad_norm_policy_batch = 0.0
		weights_value_batch = None

		rewards = rewards.reshape(-1, self.num_agents)
		dones = dones.reshape(-1, self.num_agents)
		masks = masks.reshape(-1, 1)

		self.policy_network.rnn_hidden_state = old_rnn_hidden_state_actor.to(self.device)
		for _ in range(self.n_epochs):

			'''
			Getting the probability mass function over the action space for each agent
			'''
			# start_forward_time = time.process_time()

			dists, rnn_hidden_state_actor = self.policy_network(old_states_actor.to(self.device), old_mask_actions.to(self.device))
			probs = Categorical(dists.squeeze(0))
			logprobs = probs.log_prob(old_actions.to(self.device))

			'''
			Calculate V values
			'''
			V_values, weights_value = self.critic_network(old_states_critic.to(self.device), dists.detach(), old_one_hot_actions.to(self.device))
			V_values = V_values.reshape(-1,self.num_agents,self.num_agents)
		

			# value_loss = self.calculate_value_loss(V_values, target_V_values, weights_value)
			critic_v_loss_1 = F.mse_loss(V_values*masks.unsqueeze(-1).to(self.device), target_V_values*masks.unsqueeze(-1).to(self.device), reduction="sum") / masks.sum()
			critic_v_loss_2 = F.mse_loss(torch.clamp(V_values, V_values_old.to(self.device)-self.value_clip, V_values_old.to(self.device)+self.value_clip)*masks.unsqueeze(-1).to(self.device), target_V_values*masks.unsqueeze(-1).to(self.device), reduction="sum") / masks.sum()

			value_loss = torch.max(critic_v_loss_1, critic_v_loss_2)

			advantage, masking_advantage = self.calculate_advantages_based_on_exp(V_values_old, V_values, rewards.to(self.device), dones.to(self.device), weights_value, masks.to(self.device), episode)

			if "prd_avg" in self.experiment_type:
				agent_groups_over_episode = torch.sum(masking_advantage,dim=0)
				avg_agent_group_over_episode = torch.mean(agent_groups_over_episode.float())
			elif "threshold" in self.experiment_type:
				agent_groups_over_episode = torch.sum(torch.sum(masking_advantage.float(), dim=-2),dim=0)/masking_advantage.shape[0]
				avg_agent_group_over_episode = torch.mean(agent_groups_over_episode)
			

			# Finding the ratio (pi_theta / pi_theta__old)
			ratios = torch.exp(logprobs - old_logprobs.to(self.device))
			# Finding Surrogate Loss
			surr1 = ratios * advantage*masks.to(self.device)
			surr2 = torch.clamp(ratios, 1-self.policy_clip, 1+self.policy_clip) * advantage * masks.to(self.device)

			# final loss of clipped objective PPO
			entropy = -torch.mean(torch.sum(dists*masks.unsqueeze(-1).to(self.device) * torch.log(torch.clamp(dists*masks.unsqueeze(-1).to(self.device), 1e-10,1.0)), dim=2))
			policy_loss = (-torch.min(surr1, surr2).mean() - self.entropy_pen*entropy)


			self.critic_optimizer.zero_grad()
			value_loss.backward(retain_graph=False)
			if self.enable_grad_clip_critic:
				grad_norm_value = torch.nn.utils.clip_grad_norm_(self.critic_network.parameters(),self.grad_clip_critic)
			else:
				grad_norm_value = torch.tensor([1.0])
			# grad_norm_value = 0
			# for p in self.critic_network.parameters():
			# 	param_norm = p.grad.detach().data.norm(2)
			# 	grad_norm_value += param_norm.item() ** 2
			# grad_norm_value = torch.tensor(grad_norm_value) ** 0.5
			self.critic_optimizer.step()


			self.policy_optimizer.zero_grad()
			policy_loss.backward(retain_graph=False)
			if self.enable_grad_clip_actor:
				grad_norm_policy = torch.nn.utils.clip_grad_norm_(self.policy_network.parameters(),self.grad_clip_actor)
			else:
				grad_norm_policy = torch.tensor([1.0])
			self.policy_optimizer.step()
			self.policy_network.rnn_hidden_state = rnn_hidden_state_actor

			value_loss_batch += value_loss.item()
			policy_loss_batch += policy_loss.item()
			grad_norm_policy_batch += grad_norm_policy.item()
			grad_norm_value_batch += grad_norm_value.item()
			entropy_batch += entropy.item()
			if weights_value_batch is None:
				weights_value_batch = weights_value
			else:
				weights_value_batch += weights_value


		value_loss_batch /= self.n_epochs
		policy_loss_batch /= self.n_epochs
		grad_norm_policy_batch /= self.n_epochs
		grad_norm_value_batch /= self.n_epochs
		entropy_batch /= self.n_epochs
		weights_value_batch /= self.n_epochs


		self.update_parameters()
		self.target_policy_network.rnn_hidden_state = None
		self.policy_network.rnn_hidden_state = None

		if episode % self.network_update_interval == 0:
			# Copy new weights into old critic
			self.target_critic_network.load_state_dict(self.critic_network.state_dict())
			self.target_policy_network.load_state_dict(self.policy_network.state_dict())

		# clear buffer
		self.buffer.clear()


		self.plotting_dict = {
		"value_loss": value_loss_batch,
		"policy_loss": policy_loss_batch,
		"entropy": entropy_batch,
		"grad_norm_value":grad_norm_value_batch,
		"grad_norm_policy": grad_norm_policy_batch,
		"weights_value": weights_value_batch,
		}

		if "threshold" in self.experiment_type:
			self.plotting_dict["agent_groups_over_episode"] = agent_groups_over_episode
			self.plotting_dict["avg_agent_group_over_episode"] = avg_agent_group_over_episode
		if "prd_top" in self.experiment_type:
			self.plotting_dict["mean_min_weight_value"] = mean_min_weight_value

		if self.comet_ml is not None:
			self.plot(episode)
