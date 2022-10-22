import numpy as np
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
from torch.distributions import Categorical
from ppo_model import *
import torch.nn.functional as F

class PPOAgent:

	def __init__(
		self, 
		env, 
		dictionary,
		comet_ml,
		):

		self.env = env
		self.extension = dictionary["extension"]
		self.update_learning_rate_with_prd = dictionary["update_learning_rate_with_prd"]
		self.test_num = dictionary["test_num"]
		self.env_name = dictionary["env"]
		self.value_lr = dictionary["value_lr"]
		self.policy_lr = dictionary["policy_lr"]
		self.gamma = dictionary["gamma"]
		self.entropy_pen = dictionary["entropy_pen"]
		self.critic_weight_entropy_pen = dictionary["critic_weight_entropy_pen"]
		self.gae_lambda = dictionary["gae_lambda"]
		self.top_k = dictionary["top_k"]
		self.norm_adv = dictionary["norm_adv"]
		self.norm_returns = dictionary["norm_returns"]
		self.gif = dictionary["gif"]
		# TD lambda
		self.lambda_ = dictionary["lambda"]
		self.experiment_type = dictionary["experiment_type"]
		# Used for masking advantages above a threshold
		self.select_above_threshold = dictionary["select_above_threshold"]
		self.threshold_min = dictionary["threshold_min"]
		self.threshold_max = dictionary["threshold_max"]
		self.steps_to_take = dictionary["steps_to_take"]

		self.policy_clip = dictionary["policy_clip"]
		self.value_clip = dictionary["value_clip"]
		self.n_epochs = dictionary["n_epochs"]

		self.grad_clip_critic = dictionary["grad_clip_critic"]
		self.grad_clip_actor = dictionary["grad_clip_actor"]

		self.value_normalization = dictionary["value_normalization"]

		self.num_relevant_agents_in_relevant_set = []
		self.num_non_relevant_agents_in_relevant_set = []
		self.false_positive_rate = []
		self.num_agents = self.env.n
		self.num_actions = self.env.action_space[0].n

		# episode track
		self.episode = 0


		if "prd_above_threshold_decay" in self.experiment_type:
			self.threshold_delta = (self.select_above_threshold - self.threshold_min)/self.steps_to_take
		elif "prd_above_threshold_ascend" in self.experiment_type:
			self.threshold_delta = (self.threshold_max - self.select_above_threshold)/self.steps_to_take


		if dictionary["device"] == "gpu":
			self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		else:
			self.device = "cpu"


		self.num_teams = self.num_agents//4 # team size is 4
		self.relevant_set = torch.zeros(1,24,24).to(self.device)
		team_counter = 0
		for i in range(self.num_agents):
			if i < 4:
				self.relevant_set[0][i][:4] = torch.ones(4)
			elif i >= 4 and i < 8:
				self.relevant_set[0][i][4:8] = torch.ones(4)
			elif i >= 8 and i < 12:
				self.relevant_set[0][i][8:12] = torch.ones(4)
			elif i >= 12 and i < 16:
				self.relevant_set[0][i][12:16] = torch.ones(4)
			elif i >= 16 and i < 20:
				self.relevant_set[0][i][16:20] = torch.ones(4)
			elif i >= 20 and i < 24:
				self.relevant_set[0][i][20:24] = torch.ones(4)

		self.non_relevant_set = torch.ones(1,24,24).to(self.device) - self.relevant_set

		print("EXPERIMENT TYPE", self.experiment_type)

		# ############################### Q Network ##############################################
		obs_input_dim = 2*3+1 # crossing_team_greedy
		# obs_input_dim = 2*3 # crossing_greedy
		# obs_input_dim = 2*4 # paired_agent
		self.critic_network = Q_network(obs_input_dim=obs_input_dim, num_agents=self.num_agents, num_actions=self.num_actions, device=self.device).to(self.device)
		self.critic_network_old = Q_network(obs_input_dim=obs_input_dim, num_agents=self.num_agents, num_actions=self.num_actions, device=self.device).to(self.device)
		
		# ############################### V Network ##############################################
		# obs_input_dim = 2*3 + 1 # crossing team greedy
		# # obs_input_dim = 2*3 # crossing_greedy
		# # obs_input_dim = 2*4 # paired_agent
		# obs_output_dim = 128
		# obs_act_input_dim = obs_input_dim+self.num_actions
		# obs_act_output_dim = 128
		# final_input_dim = 128
		# final_output_dim = 1
		# self.critic_network = TransformerCritic(obs_input_dim, obs_output_dim, obs_act_input_dim, obs_act_output_dim, final_input_dim, final_output_dim, self.num_agents, self.num_actions, self.device).to(self.device)
		# self.critic_network_old = TransformerCritic(obs_input_dim, obs_output_dim, obs_act_input_dim, obs_act_output_dim, final_input_dim, final_output_dim, self.num_agents, self.num_actions, self.device).to(self.device)

		for param in self.critic_network_old.parameters():
			param.requires_grad_(False)
		# COPY
		self.critic_network_old.load_state_dict(self.critic_network.state_dict())
		
		self.seeds = [42, 142, 242, 342, 442]
		torch.manual_seed(self.seeds[dictionary["iteration"]-1])
		# POLICY
		obs_input_dim = 2*3+1 + (self.num_agents-1)*(2*2+1) # crossing_team_greedy
		# obs_input_dim = 2*3 + (self.num_agents-1)*4 # crossing_greedy
		# obs_input_dim = 2*3*self.num_agents # paired_agent
		self.policy_network = Policy(obs_input_dim=obs_input_dim, num_agents=self.num_agents, num_actions=self.num_actions, device=self.device).to(self.device)
		self.policy_network_old = Policy(obs_input_dim=obs_input_dim, num_agents=self.num_agents, num_actions=self.num_actions, device=self.device).to(self.device)
		for param in self.policy_network_old.parameters():
			param.requires_grad_(False)
		# COPY
		self.policy_network_old.load_state_dict(self.policy_network.state_dict())


		self.buffer = RolloutBuffer()


		self.greedy_policy = torch.zeros(self.num_agents,self.num_agents).to(self.device)
		for i in range(self.num_agents):
			self.greedy_policy[i][i] = 1


		if dictionary["load_models"]:
			# Loading models
			# critic_network_checkpoint = torch.load(dictionary["model_path_value"])
			# policy_network_checkpoint = torch.load(dictionary["model_path_policy"])
			if torch.cuda.is_available() is False:
				# For CPU
				self.critic_network.load_state_dict(torch.load(dictionary["model_path_value"],map_location=torch.device('cpu')))
				self.policy_network.load_state_dict(torch.load(dictionary["model_path_policy"],map_location=torch.device('cpu')))
			else:
				# For GPU
				self.critic_network.load_state_dict(torch.load(dictionary["model_path_value"]))
				self.policy_network.load_state_dict(torch.load(dictionary["model_path_policy"]))

		
		self.critic_optimizer = optim.Adam(self.critic_network.parameters(),lr=self.value_lr)
		self.policy_optimizer = optim.Adam(self.policy_network.parameters(),lr=self.policy_lr)


		# self.scheduler = optim.lr_scheduler.MultiStepLR(self.policy_optimizer, milestones=[1000], gamma=2)


		self.comet_ml = None
		if dictionary["save_comet_ml_plot"]:
			self.comet_ml = comet_ml

		self.update_time = 0.0
		self.forward_time = 0.0


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

			self.buffer.probs.append(dists.detach().cpu())
			self.buffer.logprobs.append(action_logprob.detach().cpu())

			return actions


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
		
		if self.norm_adv:
			advantages = (advantages - advantages.mean()) / advantages.std()
		
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
		
		if self.norm_returns:
			returns_tensor = (returns_tensor - returns_tensor.mean()) / returns_tensor.std()
			
		return returns_tensor


	def plot(self, episode):
		self.comet_ml.log_metric('Value_Loss',self.plotting_dict["value_loss"].item(),episode)
		self.comet_ml.log_metric('Grad_Norm_Value',self.plotting_dict["grad_norm_value"],episode)
		self.comet_ml.log_metric('Policy_Loss',self.plotting_dict["policy_loss"].item(),episode)
		self.comet_ml.log_metric('Grad_Norm_Policy',self.plotting_dict["grad_norm_policy"],episode)
		self.comet_ml.log_metric('Entropy',self.plotting_dict["entropy"].item(),episode)
		self.comet_ml.log_metric('Threshold_pred',self.plotting_dict["threshold"],episode)

		if "threshold" in self.experiment_type:
			for i in range(self.num_agents):
				agent_name = "agent"+str(i)
				self.comet_ml.log_metric('Group_Size_'+agent_name, self.plotting_dict["agent_groups_over_episode"][i].item(), episode)

			self.comet_ml.log_metric('Avg_Group_Size', self.plotting_dict["avg_agent_group_over_episode"].item(), episode)

			if "crossing_team_greedy" == self.env_name:
				self.comet_ml.log_metric('Num_relevant_agents_in_relevant_set',torch.mean(self.plotting_dict["num_relevant_agents_in_relevant_set"]),episode)
				self.comet_ml.log_metric('Num_non_relevant_agents_in_relevant_set',torch.mean(self.plotting_dict["num_non_relevant_agents_in_relevant_set"]),episode)
				self.num_relevant_agents_in_relevant_set.append(torch.mean(self.plotting_dict["num_relevant_agents_in_relevant_set"]).item())
				self.num_non_relevant_agents_in_relevant_set.append(torch.mean(self.plotting_dict["num_non_relevant_agents_in_relevant_set"]).item())
				# FPR = FP / (FP+TN)
				FP = torch.mean(self.plotting_dict["num_non_relevant_agents_in_relevant_set"]).item()*self.num_agents
				TN = torch.mean(self.plotting_dict["true_negatives"]).item()*self.num_agents
				self.false_positive_rate.append(FP/(FP+TN))

		if "prd_top" in self.experiment_type:
			self.comet_ml.log_metric('Mean_Smallest_Weight', self.plotting_dict["mean_min_weight_value"].item(), episode)


		# ENTROPY OF WEIGHTS
		entropy_weights = -torch.mean(torch.sum(self.plotting_dict["weights_value"]* torch.log(torch.clamp(self.plotting_dict["weights_value"], 1e-10,1.0)), dim=2))
		self.comet_ml.log_metric('Critic_Weight_Entropy', entropy_weights.item(), episode)


	def calculate_advantages_based_on_exp(self, V_values, rewards, dones, weights_prd, episode):
		advantage = None
		masking_advantage = None
		mean_min_weight_value = -1
		if "shared" in self.experiment_type:
			advantage = torch.sum(self.calculate_advantages(V_values, rewards, dones),dim=-2)
		elif "prd_above_threshold" in self.experiment_type:
			masking_advantage = (weights_prd>self.select_above_threshold).int()
			advantage = torch.sum(self.calculate_advantages(V_values, rewards, dones) * torch.transpose(masking_advantage,-1,-2),dim=-2)
		elif "top" in self.experiment_type:
			if episode < self.steps_to_take:
				advantage = torch.sum(self.calculate_advantages(V_values, rewards, dones),dim=-2)
				masking_advantage = torch.ones(weights_prd.shape).to(self.device)
				min_weight_values, _ = torch.min(weights_prd, dim=-1)
				mean_min_weight_value = torch.mean(min_weight_values)
			else:
				values, indices = torch.topk(weights_prd,k=self.top_k,dim=-1)
				min_weight_values, _ = torch.min(values, dim=-1)
				mean_min_weight_value = torch.mean(min_weight_values)
				masking_advantage = torch.sum(F.one_hot(indices, num_classes=self.num_agents), dim=-2)
				advantage = torch.sum(self.calculate_advantages(V_values, rewards, dones) * torch.transpose(masking_advantage,-1,-2),dim=-2)
		elif "prd_soft_adv" in self.experiment_type:
			if episode < self.steps_to_take:
				advantage = torch.sum(self.calculate_advantages(V_values, rewards, dones),dim=-2)
			else:
				advantage = torch.sum(self.calculate_advantages(V_values, rewards, dones) * torch.transpose(weights_prd,-1,-2) ,dim=-2)
		elif "greedy" in self.experiment_type:
			advantage = torch.sum(self.calculate_advantages(V_values, rewards, dones) * self.greedy_policy ,dim=-2)
		elif "relevant_set" in self.experiment_type:
			advantage = torch.sum(self.calculate_advantages(V_values, rewards, dones) * self.relevant_set ,dim=-2)

		if "scaled" in self.experiment_type and episode > self.steps_to_take and "top" in self.experiment_type:
			advantage = advantage*(self.num_agents/self.top_k)

		return advantage, masking_advantage, mean_min_weight_value

	def update_parameters(self):
		# increment episode
		self.episode+=1

		if self.select_above_threshold > self.threshold_min and "prd_above_threshold_decay" in self.experiment_type:
			self.select_above_threshold = self.select_above_threshold - self.threshold_delta

		if self.threshold_max >= self.select_above_threshold and "prd_above_threshold_ascend" in self.experiment_type:
			self.select_above_threshold = self.select_above_threshold + self.threshold_delta


	def update(self,episode):
		# convert list to tensor
		old_states_critic = torch.FloatTensor(np.array(self.buffer.states_critic)).to(self.device)
		old_states_actor = torch.FloatTensor(np.array(self.buffer.states_actor)).to(self.device)
		old_actions = torch.FloatTensor(np.array(self.buffer.actions)).to(self.device)
		old_one_hot_actions = torch.FloatTensor(np.array(self.buffer.one_hot_actions)).to(self.device)
		old_probs = torch.stack(self.buffer.probs, dim=0).to(self.device)
		old_logprobs = torch.stack(self.buffer.logprobs, dim=0).to(self.device)
		rewards = torch.FloatTensor(np.array(self.buffer.rewards)).to(self.device)
		dones = torch.FloatTensor(np.array(self.buffer.dones)).long().to(self.device)


		Values_old, Q_values_old, weights_value_old, scores_old = self.critic_network_old(old_states_critic, old_probs.squeeze(-2), old_one_hot_actions)
		Values_old = Values_old.reshape(-1,self.num_agents,self.num_agents)

		if self.value_normalization:
			Q_values_old = torch.sum(self.critic_network_old.pop_art.denormalize(Q_values_old)*old_one_hot_actions, dim=-1).unsqueeze(-1)
		
		Q_value_target = self.nstep_returns(Q_values_old, rewards, dones).detach()

		value_loss_batch = 0
		policy_loss_batch = 0
		entropy_batch = 0
		value_weights_batch = None
		grad_norm_value_batch = 0
		grad_norm_policy_batch = 0
		agent_groups_over_episode_batch = 0
		avg_agent_group_over_episode_batch = 0
		threshold_batch = 0

		# torch.autograd.set_detect_anomaly(True)
		# Optimize policy for n epochs
		for _ in range(self.n_epochs):

			Value, Q_value, weights_value, scores = self.critic_network(old_states_critic, old_probs.squeeze(-2), old_one_hot_actions)
			Value = Value.reshape(-1,self.num_agents,self.num_agents)

			advantage, masking_advantage, mean_min_weight_value = self.calculate_advantages_based_on_exp(Value, rewards, dones, weights_value, episode)

			if "threshold" in self.experiment_type:
				agent_groups_over_episode = torch.sum(torch.sum(masking_advantage.float(), dim=-2),dim=0)/masking_advantage.shape[0]
				avg_agent_group_over_episode = torch.mean(agent_groups_over_episode)
				agent_groups_over_episode_batch += agent_groups_over_episode
				avg_agent_group_over_episode_batch += avg_agent_group_over_episode

			dists = self.policy_network(old_states_actor)
			probs = Categorical(dists.squeeze(0))
			logprobs = probs.log_prob(old_actions)

			if self.value_normalization:
				self.critic_network.pop_art.update(Q_value_target)
				Q_value_target_normalized = torch.sum(self.critic_network.pop_art.normalize(Q_value_target)*old_one_hot_actions, dim=-1).unsqueeze(-1) # gives for all possible actions
				critic_loss_1 = F.smooth_l1_loss(Q_value,Q_value_target_normalized)
				critic_loss_2 = F.smooth_l1_loss(torch.clamp(Q_value, Q_values_old-self.value_clip, Q_values_old+self.value_clip),Q_value_target_normalized)
			else:
				critic_loss_1 = F.smooth_l1_loss(Q_value,Q_value_target)
				critic_loss_2 = F.smooth_l1_loss(torch.clamp(Q_value, Q_values_old-self.value_clip, Q_values_old+self.value_clip),Q_value_target)


			# Finding the ratio (pi_theta / pi_theta__old)
			ratios = torch.exp(logprobs - old_logprobs)
			# Finding Surrogate Loss
			surr1 = ratios * advantage.detach()
			surr2 = torch.clamp(ratios, 1-self.policy_clip, 1+self.policy_clip) * advantage.detach()

			# final loss of clipped objective PPO
			entropy = -torch.mean(torch.sum(dists * torch.log(torch.clamp(dists, 1e-10,1.0)), dim=2))
			policy_loss = -torch.min(surr1, surr2).mean() - self.entropy_pen*entropy
			
			entropy_weights = -torch.mean(torch.sum(weights_value* torch.log(torch.clamp(weights_value, 1e-10,1.0)), dim=2))
			critic_loss = torch.max(critic_loss_1, critic_loss_2) + self.critic_weight_entropy_pen*entropy_weights
			

			# take gradient step
			self.critic_optimizer.zero_grad()
			critic_loss.backward()
			grad_norm_value = torch.nn.utils.clip_grad_norm_(self.critic_network.parameters(),self.grad_clip_critic)
			self.critic_optimizer.step()

			self.policy_optimizer.zero_grad()
			policy_loss.backward()
			grad_norm_policy = torch.nn.utils.clip_grad_norm_(self.policy_network.parameters(),self.grad_clip_actor)
			self.policy_optimizer.step()

			value_loss_batch += critic_loss
			policy_loss_batch += policy_loss
			entropy_batch += entropy
			grad_norm_value_batch += grad_norm_value
			grad_norm_policy_batch += grad_norm_policy
			if value_weights_batch is None:
				value_weights_batch = torch.zeros_like(weights_value.cpu())
			value_weights_batch += weights_value.detach().cpu()


			
		# Copy new weights into old policy
		self.policy_network_old.load_state_dict(self.policy_network.state_dict())

		# Copy new weights into old critic
		self.critic_network_old.load_state_dict(self.critic_network.state_dict())

		# self.scheduler.step()
		# print("learning rate of policy", self.scheduler.get_lr())

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
		threshold_batch /= self.n_epochs

		if "prd" in self.experiment_type and "prd_soft" not in self.experiment_type and "crossing_team_greedy" == self.env_name:
			num_relevant_agents_in_relevant_set = self.relevant_set*masking_advantage
			num_non_relevant_agents_in_relevant_set = self.non_relevant_set*masking_advantage
			true_negatives = self.non_relevant_set*(1-masking_advantage)
		else:
			num_relevant_agents_in_relevant_set = None
			num_non_relevant_agents_in_relevant_set = None
			true_negatives = None

		if "prd" in self.experiment_type and self.update_learning_rate_with_prd:
			for g in self.policy_optimizer.param_groups:
				g['lr'] = self.policy_lr * self.num_agents/avg_agent_group_over_episode_batch

		self.update_parameters()


		self.plotting_dict = {
		"value_loss": value_loss_batch,
		"policy_loss": policy_loss_batch,
		"entropy": entropy_batch,
		"grad_norm_value":grad_norm_value_batch,
		"grad_norm_policy": grad_norm_policy_batch,
		"weights_value": value_weights_batch,
		"threshold": threshold_batch,
		"num_relevant_agents_in_relevant_set": num_relevant_agents_in_relevant_set,
		"num_non_relevant_agents_in_relevant_set": num_non_relevant_agents_in_relevant_set,
		"true_negatives": true_negatives,
		}

		if "threshold" in self.experiment_type:
			self.plotting_dict["agent_groups_over_episode"] = agent_groups_over_episode_batch
			self.plotting_dict["avg_agent_group_over_episode"] = avg_agent_group_over_episode_batch
		if "prd_top" in self.experiment_type:
			self.plotting_dict["mean_min_weight_value"] = mean_min_weight_value

		if self.comet_ml is not None:
			self.plot(episode)

	def V_update(self, episode):

		# convert list to tensor
		old_states_critic = torch.FloatTensor(np.array(self.buffer.states_critic)).to(self.device)
		old_states_actor = torch.FloatTensor(np.array(self.buffer.states_actor)).to(self.device)
		old_actions = torch.FloatTensor(np.array(self.buffer.actions)).to(self.device)
		old_one_hot_actions = torch.FloatTensor(np.array(self.buffer.one_hot_actions)).to(self.device)
		old_probs = torch.stack(self.buffer.probs, dim=0).to(self.device)
		old_logprobs = torch.stack(self.buffer.logprobs, dim=0).to(self.device)
		rewards = torch.FloatTensor(np.array(self.buffer.rewards)).to(self.device)
		dones = torch.FloatTensor(np.array(self.buffer.dones)).long().to(self.device)

		'''
		Getting the probability mass function over the action space for each agent
		'''

		'''
		Calculate V values
		'''

		V_values_old, weights_value_old = self.critic_network_old.forward(old_states_critic, old_probs.squeeze(-2), old_one_hot_actions)
		V_values_old = V_values_old.reshape(-1,self.num_agents,self.num_agents)

		target_V_values = self.nstep_returns(V_values_old, rewards, dones).detach()


		value_loss_batch = 0
		policy_loss_batch = 0
		entropy_batch = 0
		value_weights_batch = None
		grad_norm_value_batch = 0
		grad_norm_policy_batch = 0
		agent_groups_over_episode_batch = 0
		avg_agent_group_over_episode_batch = 0
		threshold_batch = 0

		# torch.autograd.set_detect_anomaly(True)
		# Optimize policy for n epochs
		for _ in range(self.n_epochs):

			Value, weights_value = self.critic_network(old_states_critic, old_probs.squeeze(-2), old_one_hot_actions)
			Value = Value.reshape(-1,self.num_agents,self.num_agents)


			advantage, masking_advantage, mean_min_weight_value = self.calculate_advantages_based_on_exp(Value, rewards, dones, weights_value, episode)

			if "threshold" in self.experiment_type:
				agent_groups_over_episode = torch.sum(torch.sum(masking_advantage.float(), dim=-2),dim=0)/masking_advantage.shape[0]
				avg_agent_group_over_episode = torch.mean(agent_groups_over_episode)
				agent_groups_over_episode_batch += agent_groups_over_episode
				avg_agent_group_over_episode_batch += avg_agent_group_over_episode

			dists = self.policy_network(old_states_actor)
			probs = Categorical(dists.squeeze(0))
			logprobs = probs.log_prob(old_actions)

			critic_loss_1 = F.smooth_l1_loss(Value,target_V_values)
			critic_loss_2 = F.smooth_l1_loss(torch.clamp(Value, V_values_old-self.value_clip, V_values_old+self.value_clip),target_V_values)


			# Finding the ratio (pi_theta / pi_theta__old)
			ratios = torch.exp(logprobs - old_logprobs)
			# Finding Surrogate Loss
			surr1 = ratios * advantage.detach()
			surr2 = torch.clamp(ratios, 1-self.policy_clip, 1+self.policy_clip) * advantage.detach()

			# final loss of clipped objective PPO
			entropy = -torch.mean(torch.sum(dists * torch.log(torch.clamp(dists, 1e-10,1.0)), dim=2))
			policy_loss = -torch.min(surr1, surr2).mean() - self.entropy_pen*entropy
			
			entropy_weights = -torch.mean(torch.sum(weights_value* torch.log(torch.clamp(weights_value, 1e-10,1.0)), dim=2))
			critic_loss = torch.max(critic_loss_1, critic_loss_2) + self.critic_weight_entropy_pen*entropy_weights
			

			# take gradient step
			self.critic_optimizer.zero_grad()
			critic_loss.backward()
			grad_norm_value = torch.nn.utils.clip_grad_norm_(self.critic_network.parameters(),self.grad_clip_critic)
			self.critic_optimizer.step()

			self.policy_optimizer.zero_grad()
			policy_loss.backward()
			grad_norm_policy = torch.nn.utils.clip_grad_norm_(self.policy_network.parameters(),self.grad_clip_actor)
			self.policy_optimizer.step()

			value_loss_batch += critic_loss
			policy_loss_batch += policy_loss
			entropy_batch += entropy
			grad_norm_value_batch += grad_norm_value
			grad_norm_policy_batch += grad_norm_policy
			if value_weights_batch is None:
				value_weights_batch = torch.zeros_like(weights_value.cpu())
			value_weights_batch += weights_value.detach().cpu()


			
		# Copy new weights into old policy
		self.policy_network_old.load_state_dict(self.policy_network.state_dict())

		# Copy new weights into old critic
		self.critic_network_old.load_state_dict(self.critic_network.state_dict())

		# self.scheduler.step()
		# print("learning rate of policy", self.scheduler.get_lr())

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
		threshold_batch /= self.n_epochs

		if "prd" in self.experiment_type and "prd_soft" not in self.experiment_type and "crossing_team_greedy" == self.env_name:
			num_relevant_agents_in_relevant_set = self.relevant_set*masking_advantage
			num_non_relevant_agents_in_relevant_set = self.non_relevant_set*masking_advantage
			true_negatives = self.non_relevant_set*(1-masking_advantage)
		else:
			num_relevant_agents_in_relevant_set = None
			num_non_relevant_agents_in_relevant_set = None
			true_negatives = None

		if "prd" in self.experiment_type and self.update_learning_rate_with_prd:
			for g in self.policy_optimizer.param_groups:
				g['lr'] = self.policy_lr * self.num_agents/avg_agent_group_over_episode_batch

		self.update_parameters()


		self.plotting_dict = {
		"value_loss": value_loss_batch,
		"policy_loss": policy_loss_batch,
		"entropy": entropy_batch,
		"grad_norm_value":grad_norm_value_batch,
		"grad_norm_policy": grad_norm_policy_batch,
		"weights_value": value_weights_batch,
		"threshold": threshold_batch,
		"num_relevant_agents_in_relevant_set": num_relevant_agents_in_relevant_set,
		"num_non_relevant_agents_in_relevant_set": num_non_relevant_agents_in_relevant_set,
		"true_negatives": true_negatives,
		}

		if "threshold" in self.experiment_type:
			self.plotting_dict["agent_groups_over_episode"] = agent_groups_over_episode_batch
			self.plotting_dict["avg_agent_group_over_episode"] = avg_agent_group_over_episode_batch
		if "prd_top" in self.experiment_type:
			self.plotting_dict["mean_min_weight_value"] = mean_min_weight_value

		if self.comet_ml is not None:
			self.plot(episode)


	def a2c_update(self,episode):
		# convert list to tensor
		old_states_critic = torch.FloatTensor(np.array(self.buffer.states_critic)).to(self.device)
		old_states_actor = torch.FloatTensor(np.array(self.buffer.states_actor)).to(self.device)
		old_actions = torch.FloatTensor(np.array(self.buffer.actions)).to(self.device)
		old_one_hot_actions = torch.FloatTensor(np.array(self.buffer.one_hot_actions)).to(self.device)
		old_probs = torch.stack(self.buffer.probs, dim=0).to(self.device)
		old_logprobs = torch.stack(self.buffer.logprobs, dim=0).to(self.device)
		rewards = torch.FloatTensor(np.array(self.buffer.rewards)).to(self.device)
		dones = torch.FloatTensor(np.array(self.buffer.dones)).long().to(self.device)

		# torch.autograd.set_detect_anomaly(True)
		# Optimize policy for n epochs
		start_forward_time = time.process_time()
		# print(torch.cuda.memory_stats())
		# print("before forward pass", torch.cuda.memory_allocated()/1000000)
		dists = self.policy_network(old_states_actor)

		Value, Q_value, weights_value, scores = self.critic_network(old_states_critic, old_probs.squeeze(-2), old_one_hot_actions)
		Value = Value.reshape(-1,self.num_agents,self.num_agents)
		# print("after forward pass", torch.cuda.memory_allocated()/1000000)

		end_forward_time = time.process_time()
		self.forward_time += end_forward_time - start_forward_time

		Q_value_target = self.nstep_returns(Q_value, rewards, dones).detach()

		advantage, masking_advantage, mean_min_weight_value = self.calculate_advantages_based_on_exp(Value, rewards, dones, weights_value, episode)

		if "threshold" in self.experiment_type:
			agent_groups_over_episode = torch.sum(torch.sum(masking_advantage.float(), dim=-2),dim=0)/masking_advantage.shape[0]
			avg_agent_group_over_episode = torch.mean(agent_groups_over_episode)

		
		probs = Categorical(dists.squeeze(0))

		entropy = -torch.mean(torch.sum(dists * torch.log(torch.clamp(dists, 1e-10,1.0)), dim=2))
		policy_loss = -probs.log_prob(old_actions) * advantage.detach()
		policy_loss = policy_loss.mean() - self.entropy_pen*entropy

		entropy_weights = -torch.mean(torch.sum(weights_value* torch.log(torch.clamp(weights_value, 1e-10,1.0)), dim=2))

		if self.value_normalization:
			self.critic_network.pop_art.update(Q_value_target)
			Q_value_target_normalized = torch.sum(self.critic_network.pop_art.normalize(Q_value_target)*old_one_hot_actions, dim=-1).unsqueeze(-1) # gives for all possible actions
			critic_loss = F.smooth_l1_loss(Q_value,Q_value_target_normalized) + self.critic_weight_entropy_pen*entropy_weights
		else:
			critic_loss = F.smooth_l1_loss(Q_value,Q_value_target) + self.critic_weight_entropy_pen*entropy_weights

		if "MAAC" in self.extension:
			# regularizing the attention scores
			critic_loss += 1e-3*torch.mean(scores**2)
		

		
		start_update_time = time.process_time()

		# print("before backprop", torch.cuda.memory_allocated()/1000000)
		# take gradient step
		self.critic_optimizer.zero_grad()
		critic_loss.backward()
		grad_norm_value = torch.nn.utils.clip_grad_norm_(self.critic_network.parameters(),self.grad_clip_critic)
		self.critic_optimizer.step()


		self.policy_optimizer.zero_grad()
		policy_loss.backward()
		grad_norm_policy = torch.nn.utils.clip_grad_norm_(self.policy_network.parameters(),self.grad_clip_actor)
		self.policy_optimizer.step()

		# print("after backprop", torch.cuda.memory_allocated()/1000000)

		end_update_time = time.process_time()
		self.update_time += end_update_time - start_update_time


			
		# Copy new weights into old policy
		self.policy_network_old.load_state_dict(self.policy_network.state_dict())

		# Copy new weights into old critic
		self.critic_network_old.load_state_dict(self.critic_network.state_dict())

		# self.scheduler.step()
		# print("learning rate of policy", self.scheduler.get_lr())

		# clear buffer
		self.buffer.clear()


		if "prd" in self.experiment_type and "crossing_team_greedy" == self.env_name:
			num_relevant_agents_in_relevant_set = self.relevant_set*masking_advantage
			num_non_relevant_agents_in_relevant_set = self.non_relevant_set*masking_advantage
			true_negatives = self.non_relevant_set*(1-masking_advantage)
		else:
			num_relevant_agents_in_relevant_set = None
			num_non_relevant_agents_in_relevant_set = None
			true_negatives = None

		if "prd" in self.experiment_type and self.update_learning_rate_with_prd:
			for g in self.policy_optimizer.param_groups:
				g['lr'] = self.policy_lr * self.num_agents/avg_agent_group_over_episode_batch

		self.update_parameters()

		threshold = self.select_above_threshold
		self.plotting_dict = {
		"value_loss": critic_loss,
		"policy_loss": policy_loss,
		"entropy": entropy,
		"grad_norm_value":grad_norm_value,
		"grad_norm_policy": grad_norm_policy,
		"weights_value": weights_value,
		"threshold": threshold,
		"num_relevant_agents_in_relevant_set": num_relevant_agents_in_relevant_set,
		"num_non_relevant_agents_in_relevant_set": num_non_relevant_agents_in_relevant_set,
		"true_negatives": true_negatives,
		}

		if "threshold" in self.experiment_type:
			self.plotting_dict["agent_groups_over_episode"] = agent_groups_over_episode
			self.plotting_dict["avg_agent_group_over_episode"] = avg_agent_group_over_episode
		if "prd_top" in self.experiment_type:
			self.plotting_dict["mean_min_weight_value"] = mean_min_weight_value

		if self.comet_ml is not None:
			self.plot(episode)



	def get_policy_grad(self,episode):
		# convert list to tensor
		old_states_critic = torch.FloatTensor(np.array(self.buffer.states_critic)).to(self.device)
		old_states_actor = torch.FloatTensor(np.array(self.buffer.states_actor)).to(self.device)
		old_actions = torch.FloatTensor(np.array(self.buffer.actions)).to(self.device)
		old_one_hot_actions = torch.FloatTensor(np.array(self.buffer.one_hot_actions)).to(self.device)
		old_probs = torch.stack(self.buffer.probs, dim=0).to(self.device)
		old_logprobs = torch.stack(self.buffer.logprobs, dim=0).to(self.device)
		rewards = torch.FloatTensor(np.array(self.buffer.rewards)).to(self.device)
		dones = torch.FloatTensor(np.array(self.buffer.dones)).long().to(self.device)


		Values_old, Q_values_old, weights_value_old, scores_old = self.critic_network_old(old_states_critic, old_probs.squeeze(-2), old_one_hot_actions)
		Values_old = Values_old.reshape(-1,self.num_agents,self.num_agents)
		

		if self.value_normalization:
			Q_values_old = torch.sum(self.critic_network_old.pop_art.denormalize(Q_values_old)*old_one_hot_actions, dim=-1).unsqueeze(-1)
		
		Q_value_target = self.nstep_returns(Q_values_old, rewards, dones).detach()

		policy_grad_batch = []
		# torch.autograd.set_detect_anomaly(True)
		# Optimize policy for n epochs
		for _ in range(self.n_epochs):

			Value, Q_value, weights_value, scores = self.critic_network(old_states_critic, old_probs.squeeze(-2), old_one_hot_actions)
			Value = Value.reshape(-1,self.num_agents,self.num_agents)

			advantage, masking_advantage, mean_min_weight_value = self.calculate_advantages_based_on_exp(Value, rewards, dones, weights_value, episode)

			if "threshold" in self.experiment_type:
				agent_groups_over_episode = torch.sum(torch.sum(masking_advantage.float(), dim=-2),dim=0)/masking_advantage.shape[0]
				avg_agent_group_over_episode = torch.mean(agent_groups_over_episode)

			dists = self.policy_network(old_states_actor)
			probs = Categorical(dists.squeeze(0))
			logprobs = probs.log_prob(old_actions)

			if self.value_normalization:
				self.critic_network.pop_art.update(Q_value_target)
				Q_value_target_normalized = torch.sum(self.critic_network.pop_art.normalize(Q_value_target)*old_one_hot_actions, dim=-1).unsqueeze(-1) # gives for all possible actions
				critic_loss_1 = F.smooth_l1_loss(Q_value,Q_value_target_normalized)
				critic_loss_2 = F.smooth_l1_loss(torch.clamp(Q_value, Q_values_old-self.value_clip, Q_values_old+self.value_clip),Q_value_target_normalized)
			else:
				critic_loss_1 = F.smooth_l1_loss(Q_value,Q_value_target)
				critic_loss_2 = F.smooth_l1_loss(torch.clamp(Q_value, Q_values_old-self.value_clip, Q_values_old+self.value_clip),Q_value_target)


			# Finding the ratio (pi_theta / pi_theta__old)
			ratios = torch.exp(logprobs - old_logprobs)
			# Finding Surrogate Loss
			surr1 = ratios * advantage.detach()
			surr2 = torch.clamp(ratios, 1-self.policy_clip, 1+self.policy_clip) * advantage.detach()

			# final loss of clipped objective PPO
			entropy = -torch.mean(torch.sum(dists * torch.log(torch.clamp(dists, 1e-10,1.0)), dim=2))
			policy_loss = -torch.min(surr1, surr2).mean() - self.entropy_pen*entropy
			
			entropy_weights = -torch.mean(torch.sum(weights_value* torch.log(torch.clamp(weights_value, 1e-10,1.0)), dim=2))
			critic_loss = torch.max(critic_loss_1, critic_loss_2) + self.critic_weight_entropy_pen*entropy_weights


			# take gradient step
			# self.critic_optimizer.zero_grad()
			# critic_loss.backward()
			# grad_norm_value = torch.nn.utils.clip_grad_norm_(self.critic_network.parameters(),self.grad_clip_critic)
			# self.critic_optimizer.step()
			

			self.policy_optimizer.zero_grad()
			policy_loss.backward()
			policy_grad = []
			for name,param in self.policy_network.named_parameters():
				if param.requires_grad:
					policy_grad.append(param.grad.flatten())
			policy_grad_batch.append(torch.cat(policy_grad))
			# grad_norm_policy = torch.nn.utils.clip_grad_norm_(self.policy_network.parameters(),self.grad_clip_actor)
			# self.policy_optimizer.step()



		# clear buffer
		self.buffer.clear()

		self.update_parameters()

		return torch.mean(torch.stack(policy_grad_batch), dim=0)