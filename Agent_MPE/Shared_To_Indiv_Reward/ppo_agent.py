import numpy as np
import torch
import torch.optim as optim
from torch.distributions import Categorical
from ppo_model import *
from reward_predictor_model import *
import torch.nn.functional as F

class PPOAgent:

	def __init__(
		self, 
		env, 
		dictionary,
		comet_ml,
		):

		self.env = env
		self.policy_type = dictionary["policy_type"]
		self.critic_type = dictionary["critic_type"]
		self.test_num = dictionary["test_num"]
		self.env_name = dictionary["env"]
		self.value_lr = dictionary["value_lr"]
		self.policy_lr = dictionary["policy_lr"]
		self.reward_predictor_lr = dictionary["reward_predictor_lr"]
		self.l1_pen = dictionary["l1_pen"]
		self.l1_pen_min = dictionary["l1_pen_min"]
		self.l1_pen_steps_to_take = dictionary["l1_pen_steps_to_take"]
		self.critic_entropy_pen = dictionary["critic_entropy_pen"]
		self.gamma = dictionary["gamma"]
		self.entropy_pen = dictionary["entropy_pen"]
		self.entropy_pen_min = dictionary["entropy_pen_min"]
		self.entropy_delta = (self.entropy_pen - self.entropy_pen_min) / dictionary["max_episodes"]
		self.trace_decay = dictionary["trace_decay"]
		self.top_k = dictionary["top_k"]
		self.gae = dictionary["gae"]
		self.critic_loss_type = dictionary["critic_loss_type"]
		self.norm_adv = dictionary["norm_adv"]
		self.norm_rew = dictionary["norm_rew"]
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
		self.n_epochs = dictionary["n_epochs"]


		if "prd_above_threshold_decay" in self.experiment_type:
			self.threshold_delta = (self.select_above_threshold - self.threshold_min)/self.steps_to_take
		elif "prd_above_threshold_ascend" in self.experiment_type:
			self.threshold_delta = (self.threshold_max - self.select_above_threshold)/self.steps_to_take

		if "prd_above_threshold_l1_pen_decay" in self.experiment_type:
			self.l1_pen_delta = (self.l1_pen - self.l1_pen_min)/self.l1_pen_steps_to_take

		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		# self.device = "cpu"
		
		self.num_agents = self.env.n
		self.num_actions = self.env.action_space[0].n

		print("EXPERIMENT TYPE", self.experiment_type)

		if self.env_name == "paired_by_sharing_goals":
			obs_dim = 2*4
			# self.critic_network = TransformerCritic(obs_dim, 128, obs_dim+self.num_actions, 128, 128, 1, self.num_agents, self.num_actions).to(self.device)
		elif self.env_name == "crossing_greedy":
		# 	obs_dim = 2*3 + 2*(self.num_agents-1)
			obs_dim = 2*3
			# self.critic_network = TransformerCritic(obs_dim, 128, obs_dim+self.num_actions, 128, 128, 1, self.num_agents, self.num_actions).to(self.device)
		elif self.env_name == "crossing_fully_coop":
		# 	obs_dim = 2*3 + 2*(self.num_agents-1)
			obs_dim = 2*3
			# self.critic_network = DualTransformerCritic(obs_dim, 128, obs_dim+self.num_actions, 128, 128, 1, self.num_agents, self.num_actions).to(self.device)
		elif self.env_name == "color_social_dilemma":
			obs_dim = 2*2 + 1 + 2*3
			# self.critic_network = TransformerCritic(obs_dim, 128, obs_dim+self.num_actions, 128, 128, 1, self.num_agents, self.num_actions).to(self.device)
		elif self.env_name in ["crossing_partially_coop", "crossing_team_greedy"]:
		# 	obs_dim = 2*3 + 1 + (2+1) * (self.num_agents-1)
			obs_dim = 2*3 + 1
			# self.critic_network = DualTransformerCritic(obs_dim, 128, obs_dim+self.num_actions, 128, 128, 1, self.num_agents, self.num_actions).to(self.device)


		self.critic_network = TransformerCritic(obs_dim, 128, obs_dim+self.num_actions, 128, 128, 1, self.num_agents, self.num_actions).to(self.device)

		# self.reward_predictor = TransformerRewardPredictor(obs_dim, 128, 128, 1, self.num_agents, self.num_actions).to(self.device)
		self.reward_predictor = JointRewardPredictor(obs_dim+self.num_actions, 128, 64, 1).to(self.device)

		if self.env_name in ["paired_by_sharing_goals", "crossing_greedy", "crossing_fully_coop"]:
			obs_dim = 2*3
		elif self.env_name in ["color_social_dilemma"]:
			obs_dim = 2*2 + 1 + 2*3
		elif self.env_name in ["crossing_partially_coop", "crossing_team_greedy"]:
			obs_dim = 2*3 + 1

		# MLP POLICY
		if self.policy_type == "MLP":
			self.policy_network = MLPPolicy(obs_dim, self.num_agents, self.num_actions).to(self.device)
			self.policy_network_old = MLPPolicy(obs_dim, self.num_agents, self.num_actions).to(self.device)
		elif self.policy_type == "Transformer":
			self.policy_network = TransformerPolicy(obs_dim, 128, 128, self.num_actions, self.num_agents, self.num_actions).to(self.device)
			self.policy_network_old = TransformerPolicy(obs_dim, 128, 128, self.num_actions, self.num_agents, self.num_actions).to(self.device)

		# COPY
		self.policy_network_old.load_state_dict(self.policy_network.state_dict())

		self.buffer = RolloutBuffer()


		if self.env_name == "color_social_dilemma":
			self.relevant_set = torch.zeros(self.num_agents, self.num_agents).to(self.device)
			for i in range(self.num_agents):
				self.relevant_set[i][i] = 1
				if i < self.num_agents//2:
					for j in range(self.num_agents//2, self.num_agents):
						self.relevant_set[i][j] = 1
				else:
					for j in range(0,self.num_agents//2):
						self.relevant_set[i][j] = 1

			self.relevant_set = torch.transpose(self.relevant_set,0,1)


		self.greedy_policy = torch.zeros(self.num_agents,self.num_agents).to(self.device)
		for i in range(self.num_agents):
			self.greedy_policy[i][i] = 1


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

		
		self.critic_optimizer = optim.Adam(self.critic_network.parameters(),lr=self.value_lr)
		self.policy_optimizer = optim.Adam(self.policy_network.parameters(),lr=self.policy_lr)

		self.reward_predictor_optimizer = optim.Adam(self.reward_predictor.parameters(),lr=self.reward_predictor_lr)


		self.comet_ml = None
		if dictionary["save_comet_ml_plot"]:
			self.comet_ml = comet_ml


	def get_action(self,state):

		with torch.no_grad():
			state = torch.FloatTensor([state]).to(self.device)
			dists, _ = self.policy_network_old(state)
			actions = [Categorical(dist).sample().detach().cpu().item() for dist in dists[0]]

			probs = Categorical(dists)
			action_logprob = probs.log_prob(torch.FloatTensor(actions).to(self.device))

			self.buffer.probs.append(dists.detach().cpu())
			self.buffer.logprobs.append(action_logprob.detach().cpu())

			return actions


	def calculate_advantages(self,returns, values, rewards, dones):
		
		advantages = None

		if self.gae:
			advantages = []
			next_value = 0
			advantage = 0
			rewards = rewards.unsqueeze(-1)
			dones = dones.unsqueeze(-1)
			masks = 1 - dones
			for t in reversed(range(0, len(rewards))):
				td_error = rewards[t] + (self.gamma * next_value * masks[t]) - values.data[t]
				next_value = values.data[t]
				
				advantage = td_error + (self.gamma * self.trace_decay * advantage * masks[t])
				advantages.insert(0, advantage)

			advantages = torch.stack(advantages)	
		else:
			advantages = returns - values
		
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
		
		if self.norm_rew:
			
			returns_tensor = (returns_tensor - returns_tensor.mean()) / returns_tensor.std()
			
		return returns_tensor


	def plot(self, episode):
	
		self.comet_ml.log_metric('Value_Loss',self.plotting_dict["value_loss"].item(),episode)
		self.comet_ml.log_metric('Grad_Norm_Value',self.plotting_dict["grad_norm_value"],episode)
		self.comet_ml.log_metric('Policy_Loss',self.plotting_dict["policy_loss"].item(),episode)
		self.comet_ml.log_metric('Grad_Norm_Policy',self.plotting_dict["grad_norm_policy"],episode)
		self.comet_ml.log_metric('Entropy',self.plotting_dict["entropy"].item(),episode)
		self.comet_ml.log_metric('Reward_Loss',self.plotting_dict["reward_loss"].item(),episode)
		self.comet_ml.log_metric('Grad_Norm_Reward_Predictor',self.plotting_dict["grad_norm_reward_predictor"],episode)

		# entropy_weights = -torch.mean(torch.sum(self.plotting_dict["weight_reward_net"] * torch.log(torch.clamp(self.plotting_dict["weight_reward_net"], 1e-10,1.0)), dim=2))
		# self.comet_ml.log_metric('Reward_Predictor_Weight_Entropy', entropy_weights.item(), episode)

		entropy_weights = -torch.mean(self.plotting_dict["weight_reward_net"] * torch.log(torch.clamp(self.plotting_dict["weight_reward_net"], 1e-10,1.0)))
		self.comet_ml.log_metric('Reward_Predictor_Weight_Entropy', entropy_weights.item(), episode)

		if "threshold" in self.experiment_type:
			for i in range(self.num_agents):
				agent_name = "agent"+str(i)
				self.comet_ml.log_metric('Group_Size_'+agent_name, self.plotting_dict["agent_groups_over_episode"][i].item(), episode)

			self.comet_ml.log_metric('Avg_Group_Size', self.plotting_dict["avg_agent_group_over_episode"].item(), episode)


		if "prd_top" in self.experiment_type:
			self.comet_ml.log_metric('Mean_Smallest_Weight', self.plotting_dict["mean_min_weight_value"].item(), episode)


		if len(self.plotting_dict["weights_value"]) == 2:
			# ENTROPY OF WEIGHTS
			if "MultiHead" in self.critic_type:
				for i in range(len(self.plotting_dict["weights_value"][0])):
					entropy_weights = -torch.mean(torch.sum(self.plotting_dict["weights_value"][0][i] * torch.log(torch.clamp(self.plotting_dict["weights_value"][0][i], 1e-10,1.0)), dim=2))
					self.comet_ml.log_metric('Critic_Weight_Entropy_States', entropy_weights.item(), episode)

				for i in range(len(self.plotting_dict["weights_value"][1])):
					entropy_weights = -torch.mean(torch.sum(self.plotting_dict["weights_value"][1][i] * torch.log(torch.clamp(self.plotting_dict["weights_value"][1][i], 1e-10,1.0)), dim=2))
					self.comet_ml.log_metric('Critic_Weight_Entropy_StatesActions', entropy_weights.item(), episode)
			else:
				entropy_weights = -torch.mean(torch.sum(self.plotting_dict["weights_value"][0] * torch.log(torch.clamp(self.plotting_dict["weights_value"][0], 1e-10,1.0)), dim=2))
				self.comet_ml.log_metric('Critic_Weight_Entropy_States', entropy_weights.item(), episode)

				entropy_weights = -torch.mean(torch.sum(self.plotting_dict["weights_value"][1] * torch.log(torch.clamp(self.plotting_dict["weights_value"][1], 1e-10,1.0)), dim=2))
				self.comet_ml.log_metric('Critic_Weight_Entropy_StatesActions', entropy_weights.item(), episode)

		elif len(self.plotting_dict["weights_value"]) == 4:
			entropy_weights = -torch.mean(torch.sum(self.plotting_dict["weights_value"][0] * torch.log(torch.clamp(self.plotting_dict["weights_value"][0], 1e-10,1.0)), dim=2))
			self.comet_ml.log_metric('Critic_Weight_Entropy_States_Preproc1', entropy_weights.item(), episode)

			entropy_weights = -torch.mean(torch.sum(self.plotting_dict["weights_value"][1] * torch.log(torch.clamp(self.plotting_dict["weights_value"][1], 1e-10,1.0)), dim=2))
			self.comet_ml.log_metric('Critic_Weight_Entropy_States_Preproc2', entropy_weights.item(), episode)

			entropy_weights = -torch.mean(torch.sum(self.plotting_dict["weights_value"][2] * torch.log(torch.clamp(self.plotting_dict["weights_value"][2], 1e-10,1.0)), dim=2))
			self.comet_ml.log_metric('Critic_Weight_Entropy_States_1', entropy_weights.item(), episode)

			entropy_weights = -torch.mean(torch.sum(self.plotting_dict["weights_value"][3] * torch.log(torch.clamp(self.plotting_dict["weights_value"][3], 1e-10,1.0)), dim=2))
			self.comet_ml.log_metric('Critic_Weight_Entropy_StatesActions', entropy_weights.item(), episode)
			
		else:
			# ENTROPY OF WEIGHTS
			if "MultiHead" in self.critic_type:
				for i in range(len(self.plotting_dict["weights_value"][0])):
					entropy_weights = -torch.mean(torch.sum(self.plotting_dict["weights_value"][0][i]* torch.log(torch.clamp(self.plotting_dict["weights_value"][0][i], 1e-10,1.0)), dim=2))
					self.comet_ml.log_metric('Critic_Weight_Entropy', entropy_weights.item(), episode)
			else:
				entropy_weights = -torch.mean(torch.sum(self.plotting_dict["weights_value"][0]* torch.log(torch.clamp(self.plotting_dict["weights_value"][0], 1e-10,1.0)), dim=2))
				self.comet_ml.log_metric('Critic_Weight_Entropy', entropy_weights.item(), episode)

		
	def calculate_value_loss(self, V_values, rewards, dones, weights, weights_value, custom_loss=False):
		discounted_rewards = None
		next_probs = None

		if self.critic_loss_type == "MC":
			# we need a TxNxN vector so inflate the discounted rewards by N --> cloning the discounted rewards for an agent N times
			discounted_rewards = self.calculate_returns(rewards,self.gamma).unsqueeze(-2).repeat(1,self.num_agents,1).to(self.device)
			discounted_rewards = torch.transpose(discounted_rewards,-1,-2)
			Value_target = discounted_rewards
		elif self.critic_loss_type == "TD_1":
			next_probs, _ = self.policy_network.forward(next_states_actor)
			V_values_next, _ = self.critic_network.forward(next_states_critic, next_probs.detach(), one_hot_next_actions)
			V_values_next = V_values_next.reshape(-1,self.num_agents,self.num_agents)
			Value_target = torch.transpose(rewards.unsqueeze(-2).repeat(1,self.num_agents,1),-1,-2) + self.gamma*V_values_next*(1-dones.unsqueeze(-1))
		elif self.critic_loss_type == "TD_lambda":
			Value_target = self.nstep_returns(V_values, rewards, dones).detach()
		
		if custom_loss is False:
			value_loss = F.smooth_l1_loss(V_values, Value_target)
		else:
			weights_prd = self.calculate_prd_weights(weights_value, self.critic_network.name)
			value_loss = F.smooth_l1_loss(V_values*weights_prd, Value_target*weights_prd, reduction="sum")/V_values.shape[0]


		if self.l1_pen !=0 and self.critic_entropy_pen != 0:
			if len(weights)==2:
				if "MultiHead" in self.critic_type:
					weights_preproc = torch.mean(torch.stack(weights[0]), dim=1)
					weights_postproc = torch.mean(torch.stack(weights[1]), dim=1)
				else:
					weights_preproc = weights[0]
					weights_postproc = weights[1]

				weights_off_diagonal_preproc = weights_preproc * (1 - torch.eye(self.num_agents,device=self.device))
				weights_off_diagonal = weights_postproc * (1 - torch.eye(self.num_agents,device=self.device))
				l1_weights = torch.mean(weights_off_diagonal) + torch.mean(weights_off_diagonal_preproc)
				weight_entropy = -torch.mean(torch.sum(weights_preproc * torch.log(torch.clamp(weights_preproc, 1e-10,1.0)), dim=2)) -torch.mean(torch.sum(weights_post * torch.log(torch.clamp(weights_post, 1e-10,1.0)), dim=2))
			
			else:
				if "MultiHead" in self.critic_type:
					weights_ = torch.mean(torch.stack(weights[0]), dim=1)
				else:
					weights_ = weights


				weights_off_diagonal = weights_ * (1 - torch.eye(self.num_agents,device=self.device))
				l1_weights = torch.mean(weights_off_diagonal)
				weight_entropy = -torch.mean(torch.sum(weights_ * torch.log(torch.clamp(weights_, 1e-10,1.0)), dim=2))

			
			value_loss += self.l1_pen*l1_weights + self.critic_entropy_pen*weight_entropy

		return discounted_rewards, next_probs, value_loss


	def calculate_prd_weights(self, weights, critic_name):
		# print("weights", weights[0].shape)
		weights_prd = None
		if "MultiHeadDual" in critic_name:
			weights_ = torch.stack([weight for weight in weights[1]])
			weights_prd = torch.mean(weights_, dim=0)
		elif "MultiHead" in critic_name:
			weights_ = torch.stack([weight for weight in weights[0]])
			weights_prd = torch.mean(weights_, dim=0)
		elif "DualTransformerStateDualTransformerStateAction" in critic_name:
			weights_prd = weights[-1]
		elif "Dual" in critic_name:
			weights_prd = weights[1]
		elif "TransformerStateTransformerStateAction" in critic_name:
			weights_prd = weights[1]
		else:
			weights_prd = weights[0]

		return weights_prd


	def calculate_advantages_based_on_exp(self, discounted_rewards, V_values, rewards, dones, weights_prd, episode):
		# summing across each agent j to get the advantage
		# so we sum across the second last dimension which does A[t,j] = sum(V[t,i,j] - discounted_rewards[t,i])
		advantage = None
		masking_advantage = None
		mean_min_weight_value = -1
		if "shared" in self.experiment_type:
			advantage = torch.sum(self.calculate_advantages(discounted_rewards, V_values, rewards, dones),dim=-2)
		elif "prd_soft_adv" in self.experiment_type:
			if episode < self.steps_to_take:
				advantage = torch.sum(self.calculate_advantages(discounted_rewards, V_values, rewards, dones),dim=-2)
			else:
				advantage = torch.sum(self.calculate_advantages(discounted_rewards, V_values, rewards, dones) * torch.transpose(weights_prd,-1,-2) ,dim=-2)
		elif "prd_averaged" in self.experiment_type:
			avg_weights = torch.mean(weights_prd,dim=0)
			advantage = torch.sum(self.calculate_advantages(discounted_rewards, V_values, rewards, dones) * torch.transpose(avg_weights,-1,-2) ,dim=-2)
		elif "prd_avg_top" in self.experiment_type:
			avg_weights = torch.mean(weights_prd,dim=0)
			values, indices = torch.topk(avg_weights,k=self.top_k,dim=-1)
			masking_advantage = torch.sum(F.one_hot(indices, num_classes=self.num_agents), dim=-2)
			advantage = torch.sum(self.calculate_advantages(discounted_rewards, V_values, rewards, dones) * torch.transpose(masking_advantage,-1,-2),dim=-2)
		elif "prd_avg_above_threshold" in self.experiment_type:
			avg_weights = torch.mean(weights_prd,dim=0)
			masking_advantage = (avg_weights>self.select_above_threshold).int()
			advantage = torch.sum(self.calculate_advantages(discounted_rewards, V_values, rewards, dones) * torch.transpose(masking_advantage,-1,-2),dim=-2)
		elif "prd_above_threshold" in self.experiment_type:
			masking_advantage = (weights_prd>self.select_above_threshold).int()
			advantage = torch.sum(self.calculate_advantages(discounted_rewards, V_values, rewards, dones) * torch.transpose(masking_advantage,-1,-2),dim=-2)
		elif "top" in self.experiment_type:
			if episode < self.steps_to_take:
				advantage = torch.sum(self.calculate_advantages(discounted_rewards, V_values, rewards, dones),dim=-2)
				min_weight_values, _ = torch.min(weights_prd, dim=-1)
				mean_min_weight_value = torch.mean(min_weight_values)
			else:
				values, indices = torch.topk(weights_prd,k=self.top_k,dim=-1)
				min_weight_values, _ = torch.min(values, dim=-1)
				mean_min_weight_value = torch.mean(min_weight_values)
				masking_advantage = torch.sum(F.one_hot(indices, num_classes=self.num_agents), dim=-2)
				advantage = torch.sum(self.calculate_advantages(discounted_rewards, V_values, rewards, dones) * torch.transpose(masking_advantage,-1,-2),dim=-2)
		elif "greedy" in self.experiment_type:
			advantage = torch.sum(self.calculate_advantages(discounted_rewards, V_values, rewards, dones) * self.greedy_policy ,dim=-2)
		elif "relevant_set" in self.experiment_type:
			advantage = torch.sum(self.calculate_advantages(discounted_rewards, V_values, rewards, dones) * self.relevant_set ,dim=-2)

		if "scaled" in self.experiment_type and episode > self.steps_to_take:
			if "prd_soft_adv" in self.experiment_type:
				advantage = advantage*self.num_agents
			elif "top" in self.experiment_type:
				advantage = advantage*(self.num_agents/self.top_k)

		return advantage, masking_advantage, mean_min_weight_value

	def calculate_policy_loss(self, probs, actions, entropy, advantage):
		probs = Categorical(probs)
		policy_loss = -probs.log_prob(actions) * advantage.detach()
		policy_loss = policy_loss.mean() - self.entropy_pen*entropy

		return policy_loss

	def update_parameters(self):
		if self.select_above_threshold > self.threshold_min and "prd_above_threshold_decay" in self.experiment_type:
			self.select_above_threshold = self.select_above_threshold - self.threshold_delta

		if self.threshold_max >= self.select_above_threshold and "prd_above_threshold_ascend" in self.experiment_type:
			self.select_above_threshold = self.select_above_threshold + self.threshold_delta

		if self.l1_pen > self.l1_pen_min and "prd_above_threshold_l1_pen_decay" in self.experiment_type:
			self.l1_pen = self.l1_pen - self.l1_pen_delta

		# annealin entropy pen
		if self.entropy_pen > 0:
			self.entropy_pen = self.entropy_pen - self.entropy_delta


	def update(self,episode):

		# convert list to tensor
		old_states_critic = torch.FloatTensor(self.buffer.states_critic).to(self.device)
		old_states_actor = torch.FloatTensor(self.buffer.states_actor).to(self.device)
		old_actions = torch.FloatTensor(self.buffer.actions).to(self.device)
		old_one_hot_actions = torch.FloatTensor(self.buffer.one_hot_actions).to(self.device)
		old_probs = torch.stack(self.buffer.probs).squeeze(1).to(self.device)
		old_logprobs = torch.stack(self.buffer.logprobs).squeeze(1).to(self.device)
		indiv_rewards = torch.FloatTensor(self.buffer.indiv_rewards).to(self.device)
		shared_rewards = torch.FloatTensor(self.buffer.shared_rewards).to(self.device)
		dones = torch.FloatTensor(self.buffer.dones).to(self.device)


		'''
		Predicting indiv rewards
		'''
		# shared_reward, indiv_rewards, weight_reward_net = self.reward_predictor(states_critic)
		# reward_loss = F.smooth_l1_loss(shared_reward, shared_rewards)
		# rewards = indiv_rewards.squeeze(-1).detach()


		shared_reward, weight_reward_net = self.reward_predictor(old_states_critic, old_one_hot_actions)
		weight_reward_net = weight_reward_net.squeeze(-1)
		indiv_rewards = shared_reward*weight_reward_net

		# predicted indiv rewards to be used to train critic
		rewards = indiv_rewards


		Values_old = self.critic_network(old_states_critic, old_probs, old_one_hot_actions)
		V_values_old = Values_old[0]
		weights_value_old = Values_old[1:]
		V_values_old = V_values_old.reshape(-1,self.num_agents,self.num_agents)
		
		discounted_rewards = None
		if self.critic_loss_type == "MC":
			discounted_rewards = self.calculate_returns(rewards, dones).unsqueeze(-2).repeat(1,self.num_agents,1).to(self.device)
			Value_target = torch.transpose(discounted_rewards,-1,-2)
		elif self.critic_loss_type == "TD_lambda":
			Value_target = self.nstep_returns(V_values_old, rewards, dones).detach()

		value_loss_batch = 0
		policy_loss_batch = 0
		reward_loss_batch = 0
		entropy_batch = 0
		value_weights_batch = torch.zeros_like(weights_value_old[-1])
		policy_weights_batch = torch.zeros_like(weights_value_old[-1])
		reward_predictor_weights_batch = torch.zeros_like(weight_reward_net)
		grad_norm_value_batch = 0
		grad_norm_policy_batch = 0
		grad_norm_reward_predictor_batch = 0
		agent_groups_over_episode_batch = 0
		avg_agent_group_over_episode_batch = 0
		
		# Optimize policy for n epochs
		for _ in range(self.n_epochs):

			'''
			Predicting indiv rewards
			'''
			# shared_reward, indiv_rewards, weight_reward_net = self.reward_predictor(states_critic)
			# reward_loss = F.smooth_l1_loss(shared_reward, shared_rewards)
			# rewards = indiv_rewards.squeeze(-1).detach()


			shared_reward, weight_reward_net = self.reward_predictor(old_states_critic, old_one_hot_actions)
			weight_reward_net = weight_reward_net.squeeze(-1)
			indiv_rewards = shared_reward*weight_reward_net
			rewards = indiv_rewards
			reward_loss = F.smooth_l1_loss(shared_reward, shared_rewards)

			Value = self.critic_network(old_states_critic, old_probs, old_one_hot_actions)
			V_values = Value[0]
			weights_value = Value[1:]
			V_values = V_values.reshape(-1,self.num_agents,self.num_agents)

			if "prd" in self.experiment_type:
				weights_prd = self.calculate_prd_weights(weights_value, self.critic_network.name)
			else:
				weights_prd = None

			advantage, masking_advantage, mean_min_weight_value = self.calculate_advantages_based_on_exp(discounted_rewards, V_values, rewards, dones, weights_prd, episode)

			if "prd_avg" in self.experiment_type:
				agent_groups_over_episode = torch.sum(masking_advantage,dim=0)
				avg_agent_group_over_episode = torch.mean(agent_groups_over_episode.float())
				agent_groups_over_episode_batch += agent_groups_over_episode
				avg_agent_group_over_episode_batch += avg_agent_group_over_episode
			elif "threshold" in self.experiment_type:
				agent_groups_over_episode = torch.sum(torch.sum(masking_advantage.float(), dim=-2),dim=0)/masking_advantage.shape[0]
				avg_agent_group_over_episode = torch.mean(agent_groups_over_episode)
				agent_groups_over_episode_batch += agent_groups_over_episode
				avg_agent_group_over_episode_batch += avg_agent_group_over_episode

			# Evaluating old actions and values
			dists, weights_policy = self.policy_network(old_states_actor)
			probs = Categorical(dists)
			logprobs = probs.log_prob(old_actions)

			# Finding the ratio (pi_theta / pi_theta__old)
			ratios = torch.exp(logprobs - old_logprobs)
			# Finding Surrogate Loss
			surr1 = ratios * advantage.detach()
			surr2 = torch.clamp(ratios, 1-self.policy_clip, 1+self.policy_clip) * advantage.detach()

			# final loss of clipped objective PPO
			entropy = -torch.mean(torch.sum(dists * torch.log(torch.clamp(dists, 1e-10,1.0)), dim=2))
			policy_loss = -torch.min(surr1, surr2).mean() - self.entropy_pen*entropy


			critic_loss = F.smooth_l1_loss(V_values, Value_target)
			
			
			# take gradient step
			self.critic_optimizer.zero_grad()
			critic_loss.backward()
			grad_norm_value = torch.nn.utils.clip_grad_norm_(self.critic_network.parameters(),0.5)
			self.critic_optimizer.step()

			self.policy_optimizer.zero_grad()
			policy_loss.backward()
			grad_norm_policy = torch.nn.utils.clip_grad_norm_(self.policy_network.parameters(),0.5)
			self.policy_optimizer.step()

			self.reward_predictor_optimizer.zero_grad()
			reward_loss.backward(retain_graph=False)
			grad_norm_reward_predictor = torch.nn.utils.clip_grad_norm_(self.reward_predictor.parameters(),0.5)
			self.reward_predictor_optimizer.step()


			value_loss_batch += critic_loss
			policy_loss_batch += policy_loss
			reward_loss_batch += reward_loss
			entropy_batch += entropy
			grad_norm_value_batch += grad_norm_value
			grad_norm_policy_batch += grad_norm_policy
			grad_norm_reward_predictor_batch += grad_norm_reward_predictor
			value_weights_batch += weights_value[-1].detach()
			policy_weights_batch += weights_policy.detach()
			reward_predictor_weights_batch += weight_reward_net
			
		# Copy new weights into old policy
		self.policy_network_old.load_state_dict(self.policy_network.state_dict())

		# clear buffer
		self.buffer.clear()

		value_loss_batch /= self.n_epochs
		policy_loss_batch /= self.n_epochs
		reward_loss_batch /= self.n_epochs
		entropy_batch /= self.n_epochs
		grad_norm_value_batch /= self.n_epochs
		grad_norm_policy_batch /= self.n_epochs
		grad_norm_reward_predictor_batch /= self.n_epochs
		value_weights_batch /= self.n_epochs
		policy_weights_batch /= self.n_epochs
		reward_predictor_weights_batch /= self.n_epochs
		agent_groups_over_episode_batch /= self.n_epochs
		avg_agent_group_over_episode_batch /= self.n_epochs

		self.update_parameters()


		self.plotting_dict = {
		"value_loss": value_loss_batch,
		"policy_loss": policy_loss_batch,
		"reward_loss": reward_loss_batch,
		"entropy": entropy_batch,
		"grad_norm_value":grad_norm_value_batch,
		"grad_norm_policy": grad_norm_policy_batch,
		"grad_norm_reward_predictor": grad_norm_reward_predictor_batch,
		"weights_value": [value_weights_batch],
		"weights_policy": [policy_weights_batch],
		"weight_reward_net": reward_predictor_weights_batch,
		}

		if "threshold" in self.experiment_type:
			self.plotting_dict["agent_groups_over_episode"] = agent_groups_over_episode_batch
			self.plotting_dict["avg_agent_group_over_episode"] = avg_agent_group_over_episode_batch
		if "prd_top" in self.experiment_type:
			self.plotting_dict["mean_min_weight_value"] = mean_min_weight_value

		if self.comet_ml is not None:
			self.plot(episode)