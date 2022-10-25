import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
from torch.distributions import Categorical
from model import RolloutBuffer_Battle, RolloutBuffer_Tiger_Deer, Q_network, Policy
import torch.nn.functional as F

class Agent:

	def __init__(
		self, 
		env, 
		dictionary,
		comet_ml,
		):

		self.env = env
		self.test_num = dictionary["test_num"]
		self.env_name = dictionary["env"]
		self.value_lr = dictionary["value_lr"]
		self.policy_lr = dictionary["policy_lr"]
		self.gamma = dictionary["gamma"]
		self.entropy_pen = dictionary["entropy_pen"]
		self.gae_lambda = dictionary["gae_lambda"]
		self.gif = dictionary["gif"]
		# TD lambda
		self.lambda_ = dictionary["lambda"]
		self.experiment_type = dictionary["experiment_type"]
		# Used for masking advantages above a threshold
		self.select_above_threshold = dictionary["select_above_threshold"]

		self.policy_clip = dictionary["policy_clip"]
		self.value_clip = dictionary["value_clip"]
		self.n_epochs = dictionary["n_epochs"]

		self.grad_clip_critic = dictionary["grad_clip_critic"]
		self.grad_clip_actor = dictionary["grad_clip_actor"]
		self.attention_type = dictionary["attention_type"]

		self.num_relevant_agents_in_relevant_set = []
		self.num_non_relevant_agents_in_relevant_set = []
		self.false_positive_rate = []
		if self.env_name == "Battle":
			self.num_agents = len(self.env.agents)
		elif self.env_name == "Tiger_Deer":
			self.num_agents = sum([1 for agent in self.env.agents if "tiger" in agent])
		if self.env_name == "Battle":
			self.num_actions = 21
		elif self.env_name == "Tiger_Deer":
			self.num_actions = 9

		if dictionary["device"] == "gpu":
			self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		else:
			self.device = "cpu"


		print("EXPERIMENT TYPE", self.experiment_type)
		if self.env_name == "Battle":
			# RED AGENTS
			self.critic_network_red = Q_network(obs_input_dim=6929, num_agents=self.num_agents//2, num_actions=self.num_actions, attention_type=self.attention_type, device=self.device).to(self.device)
			self.critic_network_old_red = Q_network(obs_input_dim=6929, num_agents=self.num_agents//2, num_actions=self.num_actions, attention_type=self.attention_type, device=self.device).to(self.device)
			for param in self.critic_network_old_red.parameters():
				param.requires_grad_(False)
			# COPY
			self.critic_network_old_red.load_state_dict(self.critic_network_red.state_dict())
			# BLUE AGENTS
			self.critic_network_blue = Q_network(obs_input_dim=6929, num_agents=self.num_agents//2, num_actions=self.num_actions, attention_type=self.attention_type, device=self.device).to(self.device)
			self.critic_network_old_blue = Q_network(obs_input_dim=6929, num_agents=self.num_agents//2, num_actions=self.num_actions, attention_type=self.attention_type, device=self.device).to(self.device)
			for param in self.critic_network_old_blue.parameters():
				param.requires_grad_(False)
			# COPY
			self.critic_network_old_blue.load_state_dict(self.critic_network_blue.state_dict())
			
			self.seeds = [42, 142, 242, 342, 442]
			torch.manual_seed(self.seeds[dictionary["iteration"]-1])
			# POLICY
			# RED AGENTS
			self.policy_network_red = Policy(obs_input_dim=6929, num_agents=self.num_agents//2, num_actions=self.num_actions, device=self.device).to(self.device)
			self.policy_network_old_red = Policy(obs_input_dim=6929, num_agents=self.num_agents//2, num_actions=self.num_actions, device=self.device).to(self.device)
			for param in self.policy_network_old_red.parameters():
				param.requires_grad_(False)
			# COPY
			self.policy_network_old_red.load_state_dict(self.policy_network_red.state_dict())
			# BLUE AGENTS
			self.policy_network_blue = Policy(obs_input_dim=6929, num_agents=self.num_agents//2, num_actions=self.num_actions, device=self.device).to(self.device)
			self.policy_network_old_blue = Policy(obs_input_dim=6929, num_agents=self.num_agents//2, num_actions=self.num_actions, device=self.device).to(self.device)
			for param in self.policy_network_old_blue.parameters():
				param.requires_grad_(False)
			# COPY
			self.policy_network_old_blue.load_state_dict(self.policy_network_blue.state_dict())


			self.buffer = RolloutBuffer_Battle()

			if dictionary["load_models"]:
				# Loading models
				self.critic_network_red.load_state_dict(torch.load(dictionary["model_path_value"]))
				self.policy_network_red.load_state_dict(torch.load(dictionary["model_path_policy"]))
				self.critic_network_old_red.load_state_dict(torch.load(dictionary["model_path_value"]))
				self.policy_network_old_red.load_state_dict(torch.load(dictionary["model_path_policy"]))

				self.critic_network_blue.load_state_dict(torch.load(dictionary["model_path_value"]))
				self.policy_network_blue.load_state_dict(torch.load(dictionary["model_path_policy"]))
				self.critic_network_old_blue.load_state_dict(torch.load(dictionary["model_path_value"]))
				self.policy_network_old_blue.load_state_dict(torch.load(dictionary["model_path_policy"]))

			
			self.critic_optimizer_red = optim.Adam(self.critic_network_red.parameters(),lr=self.value_lr, weight_decay=1e-3)
			self.policy_optimizer_red = optim.Adam(self.policy_network_red.parameters(),lr=self.policy_lr, weight_decay=1e-3)

			self.critic_optimizer_blue = optim.Adam(self.critic_network_blue.parameters(),lr=self.value_lr, weight_decay=1e-3)
			self.policy_optimizer_blue = optim.Adam(self.policy_network_blue.parameters(),lr=self.policy_lr, weight_decay=1e-3)

		elif self.env_name == "Tiger_Deer":
			self.critic_network = Q_network(obs_input_dim=2349, num_agents=self.num_agents, num_actions=self.num_actions, attention_type=self.attention_type, device=self.device).to(self.device)
			self.critic_network_old = Q_network(obs_input_dim=2349, num_agents=self.num_agents, num_actions=self.num_actions, attention_type=self.attention_type, device=self.device).to(self.device)
			for param in self.critic_network_old.parameters():
				param.requires_grad_(False)
			# COPY
			self.critic_network_old.load_state_dict(self.critic_network.state_dict())
			
			self.seeds = [42, 142, 242, 342, 442]
			torch.manual_seed(self.seeds[dictionary["iteration"]-1])
			# POLICY
			self.policy_network = Policy(obs_input_dim=2349, num_agents=self.num_agents, num_actions=self.num_actions, device=self.device).to(self.device)
			self.policy_network_old = Policy(obs_input_dim=2349, num_agents=self.num_agents, num_actions=self.num_actions, device=self.device).to(self.device)
			for param in self.policy_network_old.parameters():
				param.requires_grad_(False)
			# COPY
			self.policy_network_old.load_state_dict(self.policy_network.state_dict())

			self.buffer = RolloutBuffer_Tiger_Deer()

			if dictionary["load_models"]:
				# Loading models
				self.critic_network.load_state_dict(torch.load(dictionary["model_path_value"]))
				self.policy_network.load_state_dict(torch.load(dictionary["model_path_policy"]))
				self.critic_network_old.load_state_dict(torch.load(dictionary["model_path_value"]))
				self.policy_network_old.load_state_dict(torch.load(dictionary["model_path_policy"]))

			
			self.critic_optimizer = optim.Adam(self.critic_network.parameters(),lr=self.value_lr, weight_decay=1e-3)
			self.policy_optimizer = optim.Adam(self.policy_network.parameters(),lr=self.policy_lr, weight_decay=1e-3)


		self.comet_ml = None
		if dictionary["save_comet_ml_plot"]:
			self.comet_ml = comet_ml


	def get_action(self, state_policy, agent_color, greedy=False):
		with torch.no_grad():
			state_policy = torch.FloatTensor(state_policy).to(self.device)
			if "blue" in agent_color:
				dists = self.policy_network_old_blue(state_policy)
			elif "red" in agent_color:
				dists = self.policy_network_old_red(state_policy)
			elif "tiger" in agent_color:
				dists = self.policy_network(state_policy)
			if greedy:
				# actions = [dist.argmax().detach().cpu().item() for dist in dists]
				actions = dists.argmax().detach().cpu().item()
			else:
				# actions = [Categorical(dist).sample().detach().cpu().item() for dist in dists]
				actions = Categorical(dists).sample().detach().cpu().item()

			probs = Categorical(dists)
			action_logprob = probs.log_prob(torch.FloatTensor([actions]).to(self.device))

			return actions, dists.detach().cpu().numpy(), action_logprob.detach().cpu().numpy()


	def calculate_advantages(self, values, rewards, dones):
		advantages = []
		next_value = 0
		advantage = 0
		rewards = rewards.unsqueeze(-1)
		dones = dones.unsqueeze(-1)
		masks = 1 - dones
		for t in reversed(range(0, len(rewards))):
			td_error = rewards[t] + (self.gamma * next_value * masks[t]) - values.data[t]
			next_value = values.data[t]
			
			advantage = td_error + (self.gamma * self.gae_lambda * advantage * masks[t])
			advantages.insert(0, advantage)

		advantages = torch.stack(advantages)
		
		return advantages


	def calculate_deltas(self, values, rewards, dones):
		deltas = []
		next_value = 0
		rewards = rewards.unsqueeze(-1)
		dones = dones.unsqueeze(-1)
		masks = 1-dones
		for t in reversed(range(0, len(rewards))):
			td_error = rewards[t] + (self.gamma * next_value * masks[t]) - values.data[t]
			next_value = values.data[t]
			deltas.insert(0,td_error)
		deltas = torch.stack(deltas)

		return deltas


	def nstep_returns(self,values, rewards, dones):
		deltas = self.calculate_deltas(values, rewards, dones)
		advs = self.calculate_returns(deltas, self.gamma*self.lambda_)
		target_Vs = advs+values
		return target_Vs


	def calculate_returns(self,rewards, discount_factor):
		returns = []
		R = 0
		
		for r in reversed(rewards):
			R = r + R * discount_factor
			returns.insert(0, R)
		
		returns_tensor = torch.stack(returns).to(self.device)
			
		return returns_tensor


	def plot(self, episode):
		if self.env_name == "Battle":
			self.comet_ml.log_metric('Value_Loss Red',self.plotting_dict["value_loss_red"],episode)
			self.comet_ml.log_metric('Grad_Norm_Value Red',self.plotting_dict["grad_norm_value_red"],episode)
			self.comet_ml.log_metric('Policy_Loss Red',self.plotting_dict["policy_loss_red"],episode)
			self.comet_ml.log_metric('Grad_Norm_Policy Red',self.plotting_dict["grad_norm_policy_red"],episode)
			self.comet_ml.log_metric('Entropy Red',self.plotting_dict["entropy_red"],episode)
			self.comet_ml.log_metric('Threshold_pred Red',self.plotting_dict["threshold_red"],episode)

			if "threshold" in self.experiment_type:
				for i in range(self.num_agents//2):
					agent_name = "agent"+str(i)
					self.comet_ml.log_metric('Group_Size_Red_'+agent_name, self.plotting_dict["agent_groups_over_episode_red"][i].item(), episode)

				self.comet_ml.log_metric('Avg_Group_Size_Red_', self.plotting_dict["avg_agent_group_over_episode_red"], episode)

			# ENTROPY OF WEIGHTS
			entropy_weights_red = -torch.mean(torch.sum(self.plotting_dict["weights_value_red"]* torch.log(torch.clamp(self.plotting_dict["weights_value_red"], 1e-10,1.0)), dim=2)).item()
			self.comet_ml.log_metric('Critic_Weight_Entropy_Red', entropy_weights_red, episode)


			self.comet_ml.log_metric('Value_Loss Blue',self.plotting_dict["value_loss_blue"],episode)
			self.comet_ml.log_metric('Grad_Norm_Value Blue',self.plotting_dict["grad_norm_value_blue"],episode)
			self.comet_ml.log_metric('Policy_Loss Blue',self.plotting_dict["policy_loss_blue"],episode)
			self.comet_ml.log_metric('Grad_Norm_Policy Blue',self.plotting_dict["grad_norm_policy_blue"],episode)
			self.comet_ml.log_metric('Entropy Blue',self.plotting_dict["entropy_blue"],episode)
			self.comet_ml.log_metric('Threshold_pred Blue',self.plotting_dict["threshold_blue"],episode)

			if "threshold" in self.experiment_type:
				for i in range(self.num_agents//2):
					agent_name = "agent"+str(i)
					self.comet_ml.log_metric('Group_Size_Blue_'+agent_name, self.plotting_dict["agent_groups_over_episode_blue"][i].item(), episode)

				self.comet_ml.log_metric('Avg_Group_Size_Blue_', self.plotting_dict["avg_agent_group_over_episode_blue"], episode)

			# ENTROPY OF WEIGHTS
			entropy_weights_blue = -torch.mean(torch.sum(self.plotting_dict["weights_value_blue"]* torch.log(torch.clamp(self.plotting_dict["weights_value_blue"], 1e-10,1.0)), dim=2)).item()
			self.comet_ml.log_metric('Critic_Weight_Entropy_Blue', entropy_weights_blue, episode)

		elif self.env_name == "Tiger_Deer":
			self.comet_ml.log_metric('Value_Loss',self.plotting_dict["value_loss"],episode)
			self.comet_ml.log_metric('Grad_Norm_Value',self.plotting_dict["grad_norm_value"],episode)
			self.comet_ml.log_metric('Policy_Loss',self.plotting_dict["policy_loss"],episode)
			self.comet_ml.log_metric('Grad_Norm_Policy',self.plotting_dict["grad_norm_policy"],episode)
			self.comet_ml.log_metric('Entropy',self.plotting_dict["entropy"],episode)
			self.comet_ml.log_metric('Threshold_pred',self.plotting_dict["threshold"],episode)

			if "threshold" in self.experiment_type:
				for i in range(self.num_agents):
					agent_name = "agent"+str(i)
					self.comet_ml.log_metric('Group_Size_'+agent_name, self.plotting_dict["agent_groups_over_episode"][i].item(), episode)

				self.comet_ml.log_metric('Avg_Group_Size_', self.plotting_dict["avg_agent_group_over_episode"], episode)

			# ENTROPY OF WEIGHTS
			entropy_weights = -torch.mean(torch.sum(self.plotting_dict["weights_value"]* torch.log(torch.clamp(self.plotting_dict["weights_value"], 1e-10,1.0)), dim=2)).item()
			self.comet_ml.log_metric('Critic_Weight_Entropy', entropy_weights, episode)


	def calculate_advantages_based_on_exp(self, V_values, rewards, dones, weights_prd, episode):
		advantage = None
		masking_advantage = None
		mean_min_weight_value = -1
		if "shared" in self.experiment_type:
			advantage = torch.sum(self.calculate_advantages(V_values, rewards, dones),dim=-2)
		elif "prd_above_threshold" in self.experiment_type:
			masking_advantage = (weights_prd>self.select_above_threshold).int()
			advantage = torch.sum(self.calculate_advantages(V_values, rewards, dones) * torch.transpose(masking_advantage,-1,-2),dim=-2)

		return advantage, masking_advantage

	def update(self,episode):
		if self.env_name == "Battle":
			# convert list to tensor
			old_observations_red = torch.from_numpy(np.array(self.buffer.observations_red).astype(np.float32))
			old_actions_red = torch.from_numpy(np.vstack(self.buffer.actions_red).astype(np.int))
			old_one_hot_actions_red = torch.from_numpy(np.array(self.buffer.one_hot_actions_red).astype(np.int))
			old_probs_red = torch.from_numpy(np.array(self.buffer.probs_red).astype(np.float32))
			old_logprobs_red = torch.from_numpy(np.array(self.buffer.logprobs_red).astype(np.float32)).squeeze(-1)
			rewards_red = torch.from_numpy(np.vstack(self.buffer.rewards_red).astype(np.float32))
			dones_red = torch.from_numpy(np.vstack(self.buffer.dones_red).astype(np.int)).long()

			old_observations_blue = torch.from_numpy(np.array(self.buffer.observations_blue).astype(np.float32))
			old_actions_blue = torch.from_numpy(np.vstack(self.buffer.actions_blue).astype(np.int))
			old_one_hot_actions_blue = torch.from_numpy(np.array(self.buffer.one_hot_actions_blue).astype(np.int))
			old_probs_blue = torch.from_numpy(np.array(self.buffer.probs_blue).astype(np.float32))
			old_logprobs_blue = torch.from_numpy(np.array(self.buffer.logprobs_blue).astype(np.float32)).squeeze(-1)
			rewards_blue = torch.from_numpy(np.vstack(self.buffer.rewards_blue).astype(np.float32))
			dones_blue = torch.from_numpy(np.vstack(self.buffer.dones_blue).astype(np.int)).long()


			Values_old_red, Q_values_old_red, weights_value_old_red = self.critic_network_old_red(old_observations_red.to(self.device), old_probs_red.squeeze(-2).to(self.device), old_one_hot_actions_red.to(self.device))
			Values_old_red = Values_old_red.reshape(-1,self.num_agents//2,self.num_agents//2)
			Q_value_target_red = self.nstep_returns(Q_values_old_red, rewards_red.to(self.device), dones_red.to(self.device)).detach()

			Values_old_blue, Q_values_old_blue, weights_value_old_blue = self.critic_network_old_blue(old_observations_blue.to(self.device), old_probs_blue.squeeze(-2).to(self.device), old_one_hot_actions_blue.to(self.device))
			Values_old_blue = Values_old_blue.reshape(-1,self.num_agents//2,self.num_agents//2)
			Q_value_target_blue = self.nstep_returns(Q_values_old_blue, rewards_blue.to(self.device), dones_blue.to(self.device)).detach()

			value_loss_batch_red = 0
			policy_loss_batch_red = 0
			entropy_batch_red = 0
			value_weights_batch_red = None
			grad_norm_value_batch_red = 0
			grad_norm_policy_batch_red = 0
			agent_groups_over_episode_batch_red = 0
			avg_agent_group_over_episode_batch_red = 0

			value_loss_batch_blue = 0
			policy_loss_batch_blue = 0
			entropy_batch_blue = 0
			value_weights_batch_blue = None
			grad_norm_value_batch_blue = 0
			grad_norm_policy_batch_blue = 0
			agent_groups_over_episode_batch_blue = 0
			avg_agent_group_over_episode_batch_blue = 0

			# torch.autograd.set_detect_anomaly(True)
			# Optimize policy for n epochs
			for _ in range(self.n_epochs):

				Value_red, Q_value_red, weights_value_red = self.critic_network_red(old_observations_red.to(self.device), old_probs_red.squeeze(-2).to(self.device), old_one_hot_actions_red.to(self.device))
				Value_red = Value_red.reshape(-1,self.num_agents//2,self.num_agents//2)

				Value_blue, Q_value_blue, weights_value_blue = self.critic_network_blue(old_observations_blue.to(self.device), old_probs_blue.squeeze(-2).to(self.device), old_one_hot_actions_blue.to(self.device))
				Value_blue = Value_blue.reshape(-1,self.num_agents//2,self.num_agents//2)

				advantage_red, masking_advantage_red = self.calculate_advantages_based_on_exp(Value_red, rewards_red.to(self.device), dones_red.to(self.device), weights_value_red, episode)

				advantage_blue, masking_advantage_blue = self.calculate_advantages_based_on_exp(Value_blue, rewards_blue.to(self.device), dones_blue.to(self.device), weights_value_blue, episode)

				if "threshold" in self.experiment_type:
					agent_groups_over_episode_red = torch.sum(torch.sum(masking_advantage_red.float(), dim=-2),dim=0)/masking_advantage_red.shape[0]
					avg_agent_group_over_episode_red = torch.mean(agent_groups_over_episode_red)
					agent_groups_over_episode_batch_red += agent_groups_over_episode_red
					avg_agent_group_over_episode_batch_red += avg_agent_group_over_episode_red.item()

					agent_groups_over_episode_blue = torch.sum(torch.sum(masking_advantage_blue.float(), dim=-2),dim=0)/masking_advantage_blue.shape[0]
					avg_agent_group_over_episode_blue = torch.mean(agent_groups_over_episode_blue)
					agent_groups_over_episode_batch_blue += agent_groups_over_episode_blue
					avg_agent_group_over_episode_batch_blue += avg_agent_group_over_episode_blue.item()

				dists_red = self.policy_network_red(old_observations_red.to(self.device))
				probs_red = Categorical(dists_red)
				logprobs_red = probs_red.log_prob(old_actions_red.to(self.device))

				dists_blue = self.policy_network_blue(old_observations_blue.to(self.device))
				probs_blue = Categorical(dists_blue)
				logprobs_blue = probs_blue.log_prob(old_actions_blue.to(self.device))

				critic_loss_1_red = F.smooth_l1_loss(Q_value_red,Q_value_target_red)
				critic_loss_2_red = F.smooth_l1_loss(torch.clamp(Q_value_red, Q_values_old_red-self.value_clip, Q_values_old_red+self.value_clip),Q_value_target_red)

				critic_loss_1_blue = F.smooth_l1_loss(Q_value_blue,Q_value_target_blue)
				critic_loss_2_blue = F.smooth_l1_loss(torch.clamp(Q_value_blue, Q_values_old_blue-self.value_clip, Q_values_old_blue+self.value_clip),Q_value_target_blue)


				# Finding the ratio (pi_theta / pi_theta__old)
				ratios_red = torch.exp(logprobs_red - old_logprobs_red.to(self.device))
				# Finding Surrogate Loss
				surr1_red = ratios_red * advantage_red.detach()
				surr2_red = torch.clamp(ratios_red, 1-self.policy_clip, 1+self.policy_clip) * advantage_red.detach()

				# final loss of clipped objective PPO
				entropy_red = -torch.mean(torch.sum(dists_red * torch.log(torch.clamp(dists_red, 1e-10,1.0)), dim=2))
				policy_loss_red = -torch.min(surr1_red, surr2_red).mean() - self.entropy_pen*entropy_red
				
				entropy_weights_red = -torch.mean(torch.sum(weights_value_red.detach().cpu()* torch.log(torch.clamp(weights_value_red.detach().cpu(), 1e-10,1.0)), dim=2))
				critic_loss_red = torch.max(critic_loss_1_red, critic_loss_2_red)

				# Finding the ratio (pi_theta / pi_theta__old)
				ratios_blue = torch.exp(logprobs_blue - old_logprobs_blue.to(self.device))
				# Finding Surrogate Loss
				surr1_blue = ratios_blue * advantage_blue.detach()
				surr2_blue = torch.clamp(ratios_blue, 1-self.policy_clip, 1+self.policy_clip) * advantage_blue.detach()

				# final loss of clipped objective PPO
				entropy_blue = -torch.mean(torch.sum(dists_blue * torch.log(torch.clamp(dists_blue, 1e-10,1.0)), dim=2))
				policy_loss_blue = -torch.min(surr1_blue, surr2_blue).mean() - self.entropy_pen*entropy_blue
				
				entropy_weights_blue = -torch.mean(torch.sum(weights_value_blue.detach().cpu()* torch.log(torch.clamp(weights_value_blue.detach().cpu(), 1e-10,1.0)), dim=2))
				critic_loss_blue = torch.max(critic_loss_1_blue, critic_loss_2_blue)
				

				# take gradient step
				self.critic_optimizer_red.zero_grad()
				critic_loss_red.backward()
				grad_norm_value_red = torch.nn.utils.clip_grad_norm_(self.critic_network_red.parameters(),self.grad_clip_critic)
				self.critic_optimizer_red.step()

				self.policy_optimizer_red.zero_grad()
				policy_loss_red.backward()
				grad_norm_policy_red = torch.nn.utils.clip_grad_norm_(self.policy_network_red.parameters(),self.grad_clip_actor)
				self.policy_optimizer_red.step()

				# take gradient step
				self.critic_optimizer_blue.zero_grad()
				critic_loss_blue.backward()
				grad_norm_value_blue = torch.nn.utils.clip_grad_norm_(self.critic_network_blue.parameters(),self.grad_clip_critic)
				self.critic_optimizer_blue.step()

				self.policy_optimizer_blue.zero_grad()
				policy_loss_blue.backward()
				grad_norm_policy_blue = torch.nn.utils.clip_grad_norm_(self.policy_network_blue.parameters(),self.grad_clip_actor)
				self.policy_optimizer_blue.step()

				value_loss_batch_red += critic_loss_red.item()
				policy_loss_batch_red += policy_loss_red.item()
				entropy_batch_red += entropy_red.item()
				grad_norm_value_batch_red += grad_norm_value_red.item()
				grad_norm_policy_batch_red += grad_norm_policy_red.item()
				if value_weights_batch_red is None:
					value_weights_batch_red = torch.zeros_like(weights_value_red.cpu())
				else:
					value_weights_batch_red += weights_value_red.detach().cpu()

				value_loss_batch_blue += critic_loss_blue.item()
				policy_loss_batch_blue += policy_loss_blue.item()
				entropy_batch_blue += entropy_blue.item()
				grad_norm_value_batch_blue += grad_norm_value_blue.item()
				grad_norm_policy_batch_blue += grad_norm_policy_blue.item()
				if value_weights_batch_blue is None:
					value_weights_batch_blue = torch.zeros_like(weights_value_blue.cpu())
				else:
					value_weights_batch_blue += weights_value_blue.detach().cpu()


				
			# Copy new weights into old policy
			self.policy_network_old_red.load_state_dict(self.policy_network_red.state_dict())
			self.policy_network_old_blue.load_state_dict(self.policy_network_blue.state_dict())

			# Copy new weights into old critic
			self.critic_network_old_red.load_state_dict(self.critic_network_red.state_dict())
			self.critic_network_old_blue.load_state_dict(self.critic_network_blue.state_dict())

			# clear buffer
			self.buffer.clear()

			value_loss_batch_red /= self.n_epochs
			policy_loss_batch_red /= self.n_epochs
			entropy_batch_red /= self.n_epochs
			grad_norm_value_batch_red /= self.n_epochs
			grad_norm_policy_batch_red /= self.n_epochs
			value_weights_batch_red /= self.n_epochs
			agent_groups_over_episode_batch_red /= self.n_epochs
			avg_agent_group_over_episode_batch_red /= self.n_epochs

			value_loss_batch_blue /= self.n_epochs
			policy_loss_batch_blue /= self.n_epochs
			entropy_batch_blue /= self.n_epochs
			grad_norm_value_batch_blue /= self.n_epochs
			grad_norm_policy_batch_blue /= self.n_epochs
			value_weights_batch_blue /= self.n_epochs
			agent_groups_over_episode_batch_blue /= self.n_epochs
			avg_agent_group_over_episode_batch_blue /= self.n_epochs


			self.plotting_dict = {
			"value_loss_red": value_loss_batch_red,
			"policy_loss_red": policy_loss_batch_red,
			"entropy_red": entropy_batch_red,
			"grad_norm_value_red":grad_norm_value_batch_red,
			"grad_norm_policy_red": grad_norm_policy_batch_red,
			"weights_value_red": value_weights_batch_red,
			"threshold_red": self.select_above_threshold,

			"value_loss_blue": value_loss_batch_blue,
			"policy_loss_blue": policy_loss_batch_blue,
			"entropy_blue": entropy_batch_blue,
			"grad_norm_value_blue":grad_norm_value_batch_blue,
			"grad_norm_policy_blue": grad_norm_policy_batch_blue,
			"weights_value_blue": value_weights_batch_blue,
			"threshold_blue": self.select_above_threshold
			}

			if "threshold" in self.experiment_type:
				self.plotting_dict["agent_groups_over_episode_red"] = agent_groups_over_episode_batch_red
				self.plotting_dict["avg_agent_group_over_episode_red"] = avg_agent_group_over_episode_batch_red

				self.plotting_dict["agent_groups_over_episode_blue"] = agent_groups_over_episode_batch_blue
				self.plotting_dict["avg_agent_group_over_episode_blue"] = avg_agent_group_over_episode_batch_blue

			if self.comet_ml is not None:
				self.plot(episode)

		elif self.env_name == "Tiger_Deer":
			# convert list to tensor
			# old_observations = torch.from_numpy(np.array(self.buffer.observations).astype(np.float32))
			# old_actions = torch.from_numpy(np.vstack(self.buffer.actions).astype(np.int))
			# old_one_hot_actions = torch.from_numpy(np.array(self.buffer.one_hot_actions).astype(np.int))
			# old_probs = torch.from_numpy(np.array(self.buffer.probs).astype(np.float32))
			# old_logprobs = torch.from_numpy(np.array(self.buffer.logprobs).astype(np.float32)).squeeze(-1)
			# rewards = torch.from_numpy(np.vstack(self.buffer.rewards).astype(np.float32))
			# dones = torch.from_numpy(np.vstack(self.buffer.dones).astype(np.int)).long()

			old_observations = torch.from_numpy(np.array(self.buffer.observations)).float()
			old_actions = torch.from_numpy(np.vstack(self.buffer.actions)).long()
			old_one_hot_actions = torch.from_numpy(np.array(self.buffer.one_hot_actions)).long()
			old_probs = torch.from_numpy(np.array(self.buffer.probs)).float()
			old_logprobs = torch.from_numpy(np.array(self.buffer.logprobs)).squeeze(-1).float()
			rewards = torch.from_numpy(np.vstack(self.buffer.rewards)).float()
			dones = torch.from_numpy(np.vstack(self.buffer.dones)).long()

			print(old_observations.shape, old_probs.shape, old_actions.shape, old_one_hot_actions.shape, old_logprobs.shape, rewards.shape, dones.shape)

			Values_old, Q_values_old, weights_value_old = self.critic_network_old(old_observations.to(self.device), old_probs.squeeze(-2).to(self.device), old_one_hot_actions.to(self.device))
			Values_old = Values_old.reshape(-1,self.num_agents,self.num_agents)
			Q_value_target = self.nstep_returns(Q_values_old, rewards.to(self.device), dones.to(self.device)).detach()

			value_loss_batch = 0
			policy_loss_batch = 0
			entropy_batch = 0
			value_weights_batch = None
			grad_norm_value_batch = 0
			grad_norm_policy_batch = 0
			agent_groups_over_episode_batch = 0
			avg_agent_group_over_episode_batch = 0	# torch.autograd.set_detect_anomaly(True)
			# Optimize policy for n epochs
			for _ in range(self.n_epochs):

				Value, Q_value, weights_value = self.critic_network(old_observations.to(self.device), old_probs.squeeze(-2).to(self.device), old_one_hot_actions.to(self.device))
				Value = Value.reshape(-1,self.num_agents,self.num_agents)

				advantage, masking_advantage = self.calculate_advantages_based_on_exp(Value, rewards.to(self.device), dones.to(self.device), weights_value, episode)

				if "threshold" in self.experiment_type:
					agent_groups_over_episode(torch.sum(masking_advantage.float(), dim=-2),dim=0)/masking_advantage.shape[0]
					avg_agent_group_over_episode = torch.mean(agent_groups_over_episode)
					agent_groups_over_episode_batch += agent_groups_over_episode
					avg_agent_group_over_episode_batch += avg_agent_group_over_episode.item()

				dists = self.policy_network(old_observations.to(self.device))
				probs = Categorical(dists)
				logprobs = probs.log_prob(old_actions.to(self.device))

				critic_loss_1 = F.smooth_l1_loss(Q_value,Q_value_target)
				critic_loss_2 = F.smooth_l1_loss(torch.clamp(Q_value, Q_values_old-self.value_clip, Q_values_old+self.value_clip),Q_value_target)


				# Finding the ratio (pi_theta / pi_theta__old)
				ratios = torch.exp(logprobs - old_logprobs.to(self.device))
				# Finding Surrogate Loss
				surr1 = ratios * advantage.detach()
				surr2 = torch.clamp(ratios, 1-self.policy_clip, 1+self.policy_clip) * advantage.detach()

				# final loss of clipped objective PPO
				entropy = -torch.mean(torch.sum(dists * torch.log(torch.clamp(dists, 1e-10,1.0)), dim=2))
				policy_loss = -torch.min(surr1, surr2).mean() - self.entropy_pen*entropy
				
				entropy_weights = -torch.mean(torch.sum(weights_value.detach().cpu()* torch.log(torch.clamp(weights_value.detach().cpu(), 1e-10,1.0)), dim=2))
				critic_loss = torch.max(critic_loss_1, critic_loss_2)

				
				# take gradient step
				self.critic_optimizer.zero_grad()
				critic_loss.backward()
				grad_norm_value = torch.nn.utils.clip_grad_norm_(self.critic_network.parameters(),self.grad_clip_critic)
				self.critic_optimizer.step()

				self.policy_optimizer.zero_grad()
				policy_loss.backward()	
				grad_norm_policy = torch.nn.utils.clip_grad_norm_(self.policy_network.parameters(),self.grad_clip_actor)
				self.policy_optimizer.step()

				
				value_loss_batch += critic_loss.item()
				policy_loss_batch += policy_loss.item()
				entropy_batch += entropy.item()
				grad_norm_value_batch += grad_norm_value.item()
				grad_norm_policy_batch += grad_norm_policy.item()
				if value_weights_batch is None:
					value_weights_batch = torch.zeros_like(weights_value.cpu())
				else:
					value_weights_batch += weights_value.detach().cpu()

								
			# Copy new weights into old policy
			self.policy_network_old.load_state_dict(self.policy_network.state_dict())
			self.critic_network_old.load_state_dict(self.critic_network.state_dict())

			# clear buffer
			self.buffer.clear()

			value_loss_batch /= self.n_epochs
			policy_loss_batch /= self.n_epochs
			entropy_batch /= self.n_epochs
			grad_norm_value_batch /= self.n_epochs
			grad_norm_policy_batch /= self.n_epochs
			value_weights_batch /= self.n_epochs
			agent_groups_over_episode_batch /= self.n_epochs
			avg_agent_group_over_episode_batch /= self.n_epochs


			self.plotting_dict = {
			"value_loss": value_loss_batch,
			"policy_loss": policy_loss_batch,
			"entropy": entropy_batch,
			"grad_norm_value":grad_norm_value_batch,
			"grad_norm_policy": grad_norm_policy_batch,
			"weights_value": value_weights_batch,
			"threshold": self.select_above_threshold,
			}

			if "threshold" in self.experiment_type:
				self.plotting_dict["agent_groups_over_episode"] = agent_groups_over_episode_batch
				self.plotting_dict["avg_agent_group_over_episode"] = avg_agent_group_over_episode_batch

			if self.comet_ml is not None:
				self.plot(episode)


	def a2c_update(self,episode):
		# convert list to tensor
		old_observations = torch.FloatTensor(np.array(self.buffer.observations)).to(self.device)
		old_actions = torch.FloatTensor(np.array(self.buffer.actions)).to(self.device)
		old_one_hot_actions = torch.FloatTensor(np.array(self.buffer.one_hot_actions)).to(self.device)
		old_probs = torch.stack(self.buffer.probs, dim=0).to(self.device)
		old_logprobs = torch.stack(self.buffer.logprobs, dim=0).to(self.device)
		rewards = torch.FloatTensor(np.array(self.buffer.rewards)).to(self.device)
		dones = torch.FloatTensor(np.array(self.buffer.dones)).long().to(self.device)

		# torch.autograd.set_detect_anomaly(True)
		# Optimize policy for n epochs

		Value, Q_value, weights_value = self.critic_network(old_observations, old_probs.squeeze(-2), old_one_hot_actions)
		Value = Value.reshape(-1,self.num_agents,self.num_agents)

		Q_value_target = self.nstep_returns(Q_value, rewards, dones).detach()

		advantage, masking_advantage = self.calculate_advantages_based_on_exp(Value, rewards, dones, weights_value, episode)

		if "threshold" in self.experiment_type:
			agent_groups_over_episode = torch.sum(torch.sum(masking_advantage.float(), dim=-2),dim=0)/masking_advantage.shape[0]
			avg_agent_group_over_episode = torch.mean(agent_groups_over_episode)

		dists = self.policy_network(old_observations)
		probs = Categorical(dists.squeeze(0))

		entropy = -torch.mean(torch.sum(dists * torch.log(torch.clamp(dists, 1e-10,1.0)), dim=2))
		policy_loss = -probs.log_prob(old_actions) * advantage.detach()
		policy_loss = policy_loss.mean() - self.entropy_pen*entropy

		entropy_weights = -torch.mean(torch.sum(weights_value* torch.log(torch.clamp(weights_value, 1e-10,1.0)), dim=2))

		critic_loss = F.smooth_l1_loss(Q_value,Q_value_target)
		

		# take gradient step
		self.critic_optimizer.zero_grad()
		critic_loss.backward()
		grad_norm_value = torch.nn.utils.clip_grad_norm_(self.critic_network.parameters(),self.grad_clip_critic)
		self.critic_optimizer.step()

		self.policy_optimizer.zero_grad()
		policy_loss.backward()
		grad_norm_policy = torch.nn.utils.clip_grad_norm_(self.policy_network.parameters(),self.grad_clip_actor)
		self.policy_optimizer.step()


			
		# Copy new weights into old policy
		self.policy_network_old.load_state_dict(self.policy_network.state_dict())

		# Copy new weights into old critic
		self.critic_network_old.load_state_dict(self.critic_network.state_dict())

		# clear buffer
		self.buffer.clear()


		if "prd" in self.experiment_type:
			num_relevant_agents_in_relevant_set = self.relevant_set*masking_advantage
			num_non_relevant_agents_in_relevant_set = self.non_relevant_set*masking_advantage
			true_negatives = self.non_relevant_set*(1-masking_advantage)
		else:
			num_relevant_agents_in_relevant_set = None
			num_non_relevant_agents_in_relevant_set = None
			true_negatives = None

		self.update_parameters()

		threshold = self.select_above_threshold
		self.plotting_dict = {
		"value_loss": critic_loss,
		"policy_loss": policy_loss,
		"entropy": entropy,
		"grad_norm_value":grad_norm_value,
		"grad_norm_policy": grad_norm_policy,
		"weights_value": weights_value,
		"num_relevant_agents_in_relevant_set": num_relevant_agents_in_relevant_set,
		"num_non_relevant_agents_in_relevant_set": num_non_relevant_agents_in_relevant_set,
		"true_negatives": true_negatives,
		}

		if "threshold" in self.experiment_type:
			self.plotting_dict["agent_groups_over_episode"] = agent_groups_over_episode
			self.plotting_dict["avg_agent_group_over_episode"] = avg_agent_group_over_episode

		if self.comet_ml is not None:
			self.plot(episode)