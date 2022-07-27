import numpy as np
import torch
import torch.optim as optim
from torch.distributions import Categorical
from coma_model import *
import torch.nn.functional as F

class COMAAgent:

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
		self.critic_entropy_pen = dictionary["critic_entropy_pen"]
		self.gamma = dictionary["gamma"]
		self.norm_adv = dictionary["norm_adv"]
		self.norm_rew = dictionary["norm_rew"]
		self.gif = dictionary["gif"]
		# TD lambda
		self.lambda_ = dictionary["lambda"]
		
		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		# self.device = "cpu"
		
		self.num_agents = self.env.n
		self.num_actions = self.env.action_space[0].n

		self.epsilon_start = dictionary["epsilon_start"]
		self.epsilon = self.epsilon_start
		self.epsilon_end = dictionary["epsilon_end"]
		self.epsilon_episode_steps = dictionary["epsilon_episode_steps"]
		self.episode = 0
		self.target_critic_update = dictionary["target_critic_update"]

		if self.env_name == "paired_by_sharing_goals":
			obs_dim = 2*4
			self.critic_network = TransformerCritic(obs_dim, 128, obs_dim+self.num_actions, 128, 128+128, self.num_actions, self.num_agents, self.num_actions).to(self.device)
			self.target_critic_network = TransformerCritic(obs_dim, 128, obs_dim+self.num_actions, 128, 128+128, self.num_actions, self.num_agents, self.num_actions).to(self.device)
		elif self.env_name == "crossing_greedy":
			obs_dim = 2*3
			self.critic_network = TransformerCritic(obs_dim, 128, obs_dim+self.num_actions, 128, 128+128, self.num_actions, self.num_agents, self.num_actions).to(self.device)
			self.target_critic_network = TransformerCritic(obs_dim, 128, obs_dim+self.num_actions, 128, 128+128, self.num_actions, self.num_agents, self.num_actions).to(self.device)
		elif self.env_name == "crossing_fully_coop":
			obs_dim = 2*3
			self.critic_network = DualTransformerCritic(obs_dim, 128, 128+self.num_actions, 128, 128+128, self.num_actions, self.num_agents, self.num_actions).to(self.device)
			self.target_critic_network = DualTransformerCritic(obs_dim, 128, 128+self.num_actions, 128, 128+128, self.num_actions, self.num_agents, self.num_actions).to(self.device)
		elif self.env_name == "color_social_dilemma":
			obs_dim = 2*2 + 1 + 2*3
			self.critic_network = TransformerCritic(obs_dim, 128, obs_dim+self.num_actions, 128, 128+128, self.num_actions, self.num_agents, self.num_actions).to(self.device)
			self.target_critic_network = TransformerCritic(obs_dim, 128, obs_dim+self.num_actions, 128, 128+128, self.num_actions, self.num_agents, self.num_actions).to(self.device)
		elif self.env_name in ["crossing_partially_coop", "crossing_team_greedy"]:
			obs_dim = 2*3 + 1
			self.critic_network = DualTransformerCritic(obs_dim, 128, 128+self.num_actions, 128, 128+128, self.num_actions, self.num_agents, self.num_actions).to(self.device)
			self.target_critic_network = DualTransformerCritic(obs_dim, 128, 128+self.num_actions, 128, 128+128, self.num_actions, self.num_agents, self.num_actions).to(self.device)


		self.target_critic_network.load_state_dict(self.critic_network.state_dict())

		if self.env_name in ["paired_by_sharing_goals", "crossing_greedy", "crossing_fully_coop"]:
			obs_dim = 2*3
		elif self.env_name in ["color_social_dilemma"]:
			obs_dim = 2*2 + 1 + 2*3
		elif self.env_name in ["crossing_partially_coop", "crossing_team_greedy"]:
			obs_dim = 2*3 + 1

		# MLP POLICY
		self.seeds = [42, 142, 242, 342, 442]
		torch.manual_seed(self.seeds[dictionary["iteration"]-1])
		if self.policy_type == "MLP":
			self.policy_network = MLPPolicy(obs_dim, self.num_agents, self.num_actions).to(self.device)
		elif self.policy_type == "Transformer":
			self.policy_network = TransformerPolicy(obs_dim, 128, 128, self.num_actions, self.num_agents, self.num_actions).to(self.device)


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

		# self.actor_critic_optimizer = optim.Adam(self.shared_actor_critic.parameters(),lr=self.shared_actor_critic_lr)

		self.comet_ml = None
		if dictionary["save_comet_ml_plot"]:
			self.comet_ml = comet_ml


	def get_action(self,state):
		state = torch.FloatTensor([state]).to(self.device)
		dists, _ = self.policy_network.forward(state)
		dists = (1-self.epsilon)*dists + self.epsilon/self.num_actions
		index = [Categorical(dist).sample().cpu().detach().item() for dist in dists[0]]
		return index


	def calculate_advantages(self, Q_values, baseline):
		
		advantages = Q_values - baseline
		
		if self.norm_adv:
			advantages = (advantages - advantages.mean()) / advantages.std()
		
		return advantages


	def calculate_deltas(self, values, rewards, dones):
		deltas = []
		next_value = 0
		rewards = rewards
		dones = dones
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

		
	def calculate_value_loss(self, Q_values, target_Q_values, rewards, dones, weights, weights_value, custom_loss=False):
		Q_target = self.nstep_returns(target_Q_values, rewards, dones).detach()
		
		value_loss = F.smooth_l1_loss(Q_values, Q_target)


		if self.critic_entropy_pen != 0:
			if len(weights)==2:
				if "MultiHead" in self.critic_type:
					weights_preproc = torch.mean(torch.stack(weights[0]), dim=1)
					weights_postproc = torch.mean(torch.stack(weights[1]), dim=1)
				else:
					weights_preproc = weights[0]
					weights_postproc = weights[1]

				weight_entropy = -torch.mean(torch.sum(weights_preproc * torch.log(torch.clamp(weights_preproc, 1e-10,1.0)), dim=2)) -torch.mean(torch.sum(weights_postproc * torch.log(torch.clamp(weights_postproc, 1e-10,1.0)), dim=2))
			
			else:
				if "MultiHead" in self.critic_type:
					weights_ = torch.mean(torch.stack(weights[0]), dim=1)
				else:
					weights_ = weights


				weight_entropy = -torch.mean(torch.sum(weights_ * torch.log(torch.clamp(weights_, 1e-10,1.0)), dim=2))

			
			value_loss += self.critic_entropy_pen*weight_entropy

		return value_loss



	def calculate_policy_loss(self, probs, actions, advantage):
		probs = Categorical(probs)
		policy_loss = -probs.log_prob(actions) * advantage.detach()
		policy_loss = policy_loss.mean()

		return policy_loss

	def update_parameters(self):
		self.episode += 1

		if self.epsilon>self.epsilon_end:
			self.epsilon = self.epsilon_start - self.episode*(self.epsilon_start-self.epsilon_end)/self.epsilon_episode_steps

		if self.episode%self.target_critic_update == 0:
			self.target_critic_network.load_state_dict(self.critic_network.state_dict())


	def update(self,states_critic,next_states_critic,one_hot_actions,one_hot_next_actions,actions,states_actor,next_states_actor,rewards,dones,episode):		
		'''
		Getting the probability mass function over the action space for each agent
		'''
		Policy_return = self.policy_network.forward(states_actor)
		probs = Policy_return[0]
		weights_policy = Policy_return[1:]

		Value_return = self.critic_network.forward(states_critic, one_hot_actions)
		Q_values = Value_return[0]
		weights_value = Value_return[1:]
		Q_values_act_chosen = torch.sum(Q_values.reshape(-1,self.num_agents, self.num_actions) * one_hot_actions, dim=-1)
		V_values_baseline = torch.sum(Q_values.reshape(-1,self.num_agents, self.num_actions) * probs.detach(), dim=-1)
	
		target_Value_return = self.target_critic_network.forward(states_critic, one_hot_actions)
		target_Q_values = target_Value_return[0]
		target_weights_value = target_Value_return[1:]
		target_Q_values_act_chosen = torch.sum(target_Q_values.reshape(-1,self.num_agents, self.num_actions) * one_hot_actions, dim=-1)

		value_loss = self.calculate_value_loss(Q_values_act_chosen, target_Q_values_act_chosen, rewards, dones, weights_value[-1], weights_value)

		advantage = self.calculate_advantages(Q_values_act_chosen, V_values_baseline)

		entropy = -torch.mean(torch.sum(probs * torch.log(torch.clamp(probs, 1e-10,1.0)), dim=2))
	
		policy_loss = self.calculate_policy_loss(probs, actions, advantage)
		# # ***********************************************************************************
			
		
		self.critic_optimizer.zero_grad()
		value_loss.backward(retain_graph=False)
		grad_norm_value = torch.nn.utils.clip_grad_norm_(self.critic_network.parameters(),0.5)
		self.critic_optimizer.step()


		self.policy_optimizer.zero_grad()
		policy_loss.backward(retain_graph=False)
		grad_norm_policy = torch.nn.utils.clip_grad_norm_(self.policy_network.parameters(),0.5)
		self.policy_optimizer.step()


		self.update_parameters()


		self.plotting_dict = {
		"value_loss": value_loss,
		"policy_loss": policy_loss,
		"entropy": entropy,
		"grad_norm_value":grad_norm_value,
		"grad_norm_policy": grad_norm_policy,
		"weights_value": weights_value,
		"weights_policy": weights_policy
		}

		if self.comet_ml is not None:
			self.plot(episode)