import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
from torch.distributions import Categorical
from model import RolloutBuffer, Q_network, Policy
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
		self.num_agents = len(self.env.agents)
		self.num_actions = 21

		if dictionary["device"] == "gpu":
			self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		else:
			self.device = "cpu"


		print("EXPERIMENT TYPE", self.experiment_type)

		self.critic_network = Q_network(obs_input_dim=6929, num_agents=self.num_agents, num_actions=self.num_actions, attention_type=self.attention_type, device=self.device).to(self.device)
		self.critic_network_old = Q_network(obs_input_dim=6929, num_agents=self.num_agents, num_actions=self.num_actions, attention_type=self.attention_type, device=self.device).to(self.device)
		for param in self.critic_network_old.parameters():
			param.requires_grad_(False)
		# COPY
		self.critic_network_old.load_state_dict(self.critic_network.state_dict())
		
		self.seeds = [42, 142, 242, 342, 442]
		torch.manual_seed(self.seeds[dictionary["iteration"]-1])
		# POLICY
		self.policy_network = Policy(obs_input_dim=6929, num_agents=self.num_agents, num_actions=self.num_actions, device=self.device).to(self.device)
		self.policy_network_old = Policy(obs_input_dim=6929, num_agents=self.num_agents, num_actions=self.num_actions, device=self.device).to(self.device)
		for param in self.policy_network_old.parameters():
			param.requires_grad_(False)
		# COPY
		self.policy_network_old.load_state_dict(self.policy_network.state_dict())


		self.buffer = RolloutBuffer()

		if dictionary["load_models"]:
			# Loading models
			self.critic_network.load_state_dict(torch.load(dictionary["model_path_value"]))
			self.policy_network.load_state_dict(torch.load(dictionary["model_path_policy"]))
			self.critic_network_old.load_state_dict(torch.load(dictionary["model_path_value"]))
			self.policy_network_old.load_state_dict(torch.load(dictionary["model_path_policy"]))

		
		self.critic_optimizer = optim.Adam(self.critic_network.parameters(),lr=self.value_lr, weight_decay=1e-5)
		self.policy_optimizer = optim.Adam(self.policy_network.parameters(),lr=self.policy_lr, weight_decay=1e-5)


		self.comet_ml = None
		if dictionary["save_comet_ml_plot"]:
			self.comet_ml = comet_ml


	def get_action(self, state_policy, greedy=False):
		with torch.no_grad():
			state_policy = torch.FloatTensor(state_policy).to(self.device)
			dists = self.policy_network_old(state_policy)
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

			self.comet_ml.log_metric('Num_relevant_agents_in_relevant_set',torch.mean(self.plotting_dict["num_relevant_agents_in_relevant_set"]),episode)
			self.comet_ml.log_metric('Num_non_relevant_agents_in_relevant_set',torch.mean(self.plotting_dict["num_non_relevant_agents_in_relevant_set"]),episode)
			self.num_relevant_agents_in_relevant_set.append(torch.mean(self.plotting_dict["num_relevant_agents_in_relevant_set"]).item())
			self.num_non_relevant_agents_in_relevant_set.append(torch.mean(self.plotting_dict["num_non_relevant_agents_in_relevant_set"]).item())
			# FPR = FP / (FP+TN)
			FP = torch.mean(self.plotting_dict["num_non_relevant_agents_in_relevant_set"]).item()*self.num_agents
			TN = torch.mean(self.plotting_dict["true_negatives"]).item()*self.num_agents
			self.false_positive_rate.append(FP/(FP+TN))


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

		return advantage, masking_advantage

	def update(self,episode):
		# convert list to tensor
		old_observations = torch.from_numpy(np.array(self.buffer.observations).astype(np.float32))
		old_actions = torch.from_numpy(np.vstack(self.buffer.actions).astype(np.int))
		old_one_hot_actions = torch.from_numpy(np.array(self.buffer.one_hot_actions).astype(np.int))
		old_probs = torch.from_numpy(np.array(self.buffer.probs).astype(np.float32))
		old_logprobs = torch.from_numpy(np.array(self.buffer.logprobs).astype(np.float32)).squeeze(-1)
		rewards = torch.from_numpy(np.vstack(self.buffer.rewards).astype(np.float32))
		dones = torch.from_numpy(np.vstack(self.buffer.dones).astype(np.int)).long()


		Values_old, Q_values_old, weights_value_old = self.critic_network_old(old_observations.to(self.device), old_probs.squeeze(-2).to(self.device), old_one_hot_actions.to(self.device))
		Values_old = Values_old.reshape(-1,self.num_agents,self.num_agents)
		
		Q_value_target = self.nstep_returns(Q_values_old, rewards, dones).detach()

		value_loss_batch = 0
		policy_loss_batch = 0
		entropy_batch = 0
		value_weights_batch = None
		grad_norm_value_batch = 0
		grad_norm_policy_batch = 0
		agent_groups_over_episode_batch = 0
		avg_agent_group_over_episode_batch = 0

		# torch.autograd.set_detect_anomaly(True)
		# Optimize policy for n epochs
		for _ in range(self.n_epochs):

			Value, Q_value, weights_value = self.critic_network(old_observations.to(self.device), old_probs.squeeze(-2).to(self.device), old_one_hot_actions.to(self.device))
			Value = Value.reshape(-1,self.num_agents,self.num_agents)

			advantage, masking_advantage = self.calculate_advantages_based_on_exp(Value, rewards.to(self.device), dones.to(self.device), weights_value, episode)

			if "threshold" in self.experiment_type:
				agent_groups_over_episode = torch.sum(torch.sum(masking_advantage.float(), dim=-2),dim=0)/masking_advantage.shape[0]
				avg_agent_group_over_episode = torch.mean(agent_groups_over_episode)
				agent_groups_over_episode_batch += agent_groups_over_episode
				avg_agent_group_over_episode_batch += avg_agent_group_over_episode

			dists = self.policy_network(old_observations.to(self.device))
			probs = Categorical(dists)
			logprobs = probs.log_prob(old_actions)

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

		if "prd" in self.experiment_type:
			num_relevant_agents_in_relevant_set = self.relevant_set*masking_advantage
			num_non_relevant_agents_in_relevant_set = self.non_relevant_set*masking_advantage
			true_negatives = self.non_relevant_set*(1-masking_advantage)
		else:
			num_relevant_agents_in_relevant_set = None
			num_non_relevant_agents_in_relevant_set = None
			true_negatives = None


		self.plotting_dict = {
		"value_loss": value_loss_batch,
		"policy_loss": policy_loss_batch,
		"entropy": entropy_batch,
		"grad_norm_value":grad_norm_value_batch,
		"grad_norm_policy": grad_norm_policy_batch,
		"weights_value": value_weights_batch,
		"num_relevant_agents_in_relevant_set": num_relevant_agents_in_relevant_set,
		"num_non_relevant_agents_in_relevant_set": num_non_relevant_agents_in_relevant_set,
		"true_negatives": true_negatives,
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