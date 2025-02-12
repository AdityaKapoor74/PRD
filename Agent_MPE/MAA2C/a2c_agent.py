import numpy as np
import time
import torch
import torch.optim as optim
from torch.distributions import Categorical
from a2c_model import *
import torch.nn.functional as F

class A2CAgent:

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
		self.update_after_episodes = dictionary["update_after_episodes"]
		self.network_update_interval = dictionary["network_update_interval"]
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
		if "prd_above_threshold_decay" in self.experiment_type:
			self.threshold_delta = (self.select_above_threshold - self.threshold_min)/self.steps_to_take
		elif "prd_above_threshold_ascend" in self.experiment_type:
			self.threshold_delta = (self.threshold_max - self.select_above_threshold)/self.steps_to_take

		if "prd_above_threshold_l1_pen_decay" in self.experiment_type:
			self.l1_pen_delta = (self.l1_pen - self.l1_pen_min)/self.l1_pen_steps_to_take

		if dictionary["device"] == "gpu":
			self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		else:
			self.device = "cpu"

		self.grad_clip_critic = dictionary["grad_clip_critic"]
		self.grad_clip_actor = dictionary["grad_clip_actor"]

		self.critic_observation_shape = dictionary["global_observation"]
		self.actor_observation_shape = dictionary["local_observation"]

		self.error_rate = []
		self.average_relevant_set = []
		
		self.num_agents = self.env.n
		self.num_actions = self.env.action_space[0].n
		self.episode = 0

		print("EXPERIMENT TYPE", self.experiment_type)

		# obs_input_dim = 2*3 + 1 # crossing team greedy
		# obs_input_dim = 2*3 # crossing_greedy
		# obs_input_dim = 2*4 # paired_agent
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

		# POLICY
		# obs_input_dim = 2*3+1 + (self.num_agents-1)*(2*2+1) # crossing_team_greedy
		# obs_input_dim = 2*3 + (self.num_agents-1)*4 # crossing_greedy
		# obs_input_dim = 2*3*self.num_agents # paired_agent
		self.policy_network = Policy(obs_input_dim=self.actor_observation_shape, 
			num_agents=self.num_agents, 
			num_actions=self.num_actions, 
			device=self.device).to(self.device)

		if self.env_name == "paired_by_sharing_goals":
			self.relevant_set = torch.ones(self.num_agents,self.num_agents).to(self.device)
			for i in range(self.num_agents):
				self.relevant_set[i][self.num_agents-i-1] = 0

			# here the relevant set is given value=0
			self.relevant_set = torch.transpose(self.relevant_set,0,1)
		elif self.env_name == "crossing_partially_coop":
			team_size = 8
			self.relevant_set = torch.ones(self.num_agents,self.num_agents).to(self.device)
			for i in range(self.num_agents):
				for j in range(self.num_agents):
					if i<team_size and j<team_size:
						self.relevant_set[i][j] = 0
					elif i>=team_size and i<2*team_size and j>=team_size and j<2*team_size:
						self.relevant_set[i][j] = 0
					elif i>=2*team_size and i<3*team_size and j>=2*team_size and j<3*team_size:
						self.relevant_set[i][j] = 0
					else:
						break

			# here the relevant set is given value=0
			self.relevant_set = torch.transpose(self.relevant_set,0,1)
		elif self.env_name == "crossing_team_greedy":
			team_size = 4
			self.relevant_set = torch.ones(self.num_agents,self.num_agents).to(self.device)
			for i in range(self.num_agents):
				for j in range(self.num_agents):
					if i<team_size and j<team_size:
						self.relevant_set[i][j] = 0
					elif i>=team_size and i<2*team_size and j>=team_size and j<2*team_size:
						self.relevant_set[i][j] = 0
					elif i>=2*team_size and i<3*team_size and j>=2*team_size and j<3*team_size:
						self.relevant_set[i][j] = 0
					elif i>=3*team_size and i<4*team_size and j>=3*team_size and j<4*team_size:
						self.relevant_set[i][j] = 0
					elif i>=4*team_size and i<5*team_size and j>=4*team_size and j<5*team_size:
						self.relevant_set[i][j] = 0
					elif i>=5*team_size and i<6*team_size and j>=5*team_size and j<6*team_size:
						self.relevant_set[i][j] = 0
					else:
						break


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

		
		self.critic_optimizer = optim.Adam(self.critic_network.parameters(),lr=self.value_lr, weight_decay=5e-4)
		self.policy_optimizer = optim.Adam(self.policy_network.parameters(),lr=self.policy_lr, weight_decay=5e-4)


		self.comet_ml = None
		if dictionary["save_comet_ml_plot"]:
			self.comet_ml = comet_ml


		self.update_time = 0.0
		self.forward_time = 0.0


	def get_action(self,state):
		with torch.no_grad():
			state = torch.FloatTensor(state).to(self.device)
			dists = self.policy_network.forward(state)
			index = [Categorical(dist).sample().detach().cpu().item() for dist in dists]
			return index


	def calculate_advantages(self, target_values, values, rewards, dones):
		
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
				next_value = target_values.data[t]
				
				advantage = td_error + (self.gamma * self.trace_decay * advantage * masks[t])
				advantages.insert(0, advantage)

			advantages = torch.stack(advantages)	
		else:
			advantages = target_values - values
		
		if self.norm_adv:
			advantages = (advantages - advantages.mean()) / advantages.std()
		
		return advantages


	# def calculate_deltas(self, values, rewards, dones):
	# 	deltas = []
	# 	next_value = 0
	# 	rewards = rewards.unsqueeze(-1)
	# 	dones = dones.unsqueeze(-1)
	# 	masks = 1-dones
	# 	for t in reversed(range(0, len(rewards))):
	# 		td_error = rewards[t] + (self.gamma * next_value * masks[t]) - values.data[t]
	# 		next_value = values.data[t]
	# 		deltas.insert(0,td_error)
	# 	deltas = torch.stack(deltas)

	# 	return deltas


	# def nstep_returns(self,values, rewards, dones):
	# 	deltas = self.calculate_deltas(values, rewards, dones)
	# 	advs = self.calculate_returns(deltas, self.gamma*self.lambda_)
	# 	target_Vs = advs+values
	# 	return target_Vs


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

	def build_td_lambda_targets(self, rewards, terminated, target_qs):
		# Assumes  <target_qs > in B*T*A and <reward >, <terminated >  in B*T*A, <mask > in (at least) B*T-1*1
		# Initialise  last  lambda -return  for  not  terminated  episodes
		ret = target_qs.new_zeros(*target_qs.shape)
		ret = target_qs * (1-terminated)
		# ret[:, -1] = target_qs[:, -1] * (1 - (torch.sum(terminated, dim=1)>0).int())
		# Backwards  recursive  update  of the "forward  view"
		for t in range(ret.shape[1] - 2, -1, -1):
			ret[:, t] = self.lambda_ * self.gamma * ret[:, t + 1] + \
						(rewards[:, t] + (1 - self.lambda_) * self.gamma * target_qs[:, t + 1] * (1 - terminated[:, t]))
		# Returns lambda-return from t=0 to t=T-1, i.e. in B*T-1*A
		# return ret[:, 0:-1]
		return ret


	
	def plot(self, episode):
	
		self.comet_ml.log_metric('V_Value_Loss',self.plotting_dict["value_loss"].item(),episode)
		self.comet_ml.log_metric('Grad_Norm_Value',self.plotting_dict["grad_norm_value"],episode)
		self.comet_ml.log_metric('Policy_Loss',self.plotting_dict["policy_loss"].item(),episode)
		self.comet_ml.log_metric('Grad_Norm_Policy',self.plotting_dict["grad_norm_policy"],episode)
		self.comet_ml.log_metric('Entropy',self.plotting_dict["entropy"].item(),episode)

		# if self.env_name in ["crossing_partially_coop", "paired_by_sharing_goals", "crossing_team_greedy"]:
		# 	self.comet_ml.log_metric('Relevant Set Error Rate',self.plotting_dict["relevant_set_error_rate"].item(),episode)
		# 	self.comet_ml.log_metric('Relevant Set Error Percentage',self.plotting_dict["relevant_set_error_rate"].item()*100.0,episode)
		# 	self.error_rate.append(self.plotting_dict["relevant_set_error_rate"].item())

		if "threshold" in self.experiment_type:
			for i in range(self.num_agents):
				agent_name = "agent"+str(i)
				self.comet_ml.log_metric('Group_Size_'+agent_name, self.plotting_dict["agent_groups_over_episode"][i].item(), episode)

			self.comet_ml.log_metric('Avg_Group_Size', self.plotting_dict["avg_agent_group_over_episode"].item(), episode)
			self.average_relevant_set.append(self.plotting_dict["avg_agent_group_over_episode"].item())


		if "prd_top" in self.experiment_type:
			self.comet_ml.log_metric('Mean_Smallest_Weight', self.plotting_dict["mean_min_weight_value"].item(), episode)


		entropy_weights = -torch.mean(torch.sum(self.plotting_dict["weights_value"]* torch.log(torch.clamp(self.plotting_dict["weights_value"], 1e-10,1.0)), dim=2))
		self.comet_ml.log_metric('Critic_Weight_Entropy', entropy_weights.item(), episode)

		
	def calculate_value_loss(self, V_values, target_V_values, weights):
		
		value_loss = F.smooth_l1_loss(V_values, target_V_values)

		if self.l1_pen !=0 and self.critic_entropy_pen != 0:
			weights_ = weights
			weights_off_diagonal = weights_ * (1 - torch.eye(self.num_agents,device=self.device))
			l1_weights = torch.mean(weights_off_diagonal)
			weight_entropy = -torch.mean(torch.sum(weights_ * torch.log(torch.clamp(weights_, 1e-10,1.0)), dim=2))

			value_loss += self.l1_pen*l1_weights + self.critic_entropy_pen*weight_entropy

		return value_loss


	def calculate_advantages_based_on_exp(self, target_values, V_values, rewards, dones, weights_prd, episode):
		# summing across each agent j to get the advantage
		# so we sum across the second last dimension which does A[t,j] = sum(V[t,i,j] - target_values[t,i])
		advantage = None
		masking_advantage = None
		if "shared" in self.experiment_type:
			advantage = torch.sum(self.calculate_advantages(target_values, V_values, rewards, dones),dim=-2)
		elif "prd_soft_adv" in self.experiment_type:
			if episode < self.steps_to_take:
				advantage = torch.sum(self.calculate_advantages(target_values, V_values, rewards, dones),dim=-2)
			else:
				advantage = torch.sum(self.calculate_advantages(target_values, V_values, rewards, dones) * torch.transpose(weights_prd,-1,-2) ,dim=-2)
		elif "prd_averaged" in self.experiment_type:
			avg_weights = torch.mean(weights_prd,dim=0)
			advantage = torch.sum(self.calculate_advantages(target_values, V_values, rewards, dones) * torch.transpose(avg_weights,-1,-2) ,dim=-2)
		elif "prd_avg_top" in self.experiment_type:
			avg_weights = torch.mean(weights_prd,dim=0)
			values, indices = torch.topk(avg_weights,k=self.top_k,dim=-1)
			masking_advantage = torch.sum(F.one_hot(indices, num_classes=self.num_agents), dim=-2)
			advantage = torch.sum(self.calculate_advantages(target_values, V_values, rewards, dones) * torch.transpose(masking_advantage,-1,-2),dim=-2)
		elif "prd_avg_above_threshold" in self.experiment_type:
			avg_weights = torch.mean(weights_prd,dim=0)
			masking_advantage = (avg_weights>self.select_above_threshold).int()
			advantage = torch.sum(self.calculate_advantages(target_values, V_values, rewards, dones) * torch.transpose(masking_advantage,-1,-2),dim=-2)
		elif "prd_above_threshold" in self.experiment_type:
			masking_advantage = (weights_prd>self.select_above_threshold).int()
			advantage = torch.sum(self.calculate_advantages(target_values, V_values, rewards, dones) * torch.transpose(masking_advantage,-1,-2),dim=-2)
		elif "top" in self.experiment_type:
			if episode < self.steps_to_take:
				advantage = torch.sum(self.calculate_advantages(target_values, V_values, rewards, dones),dim=-2)
				min_weight_values, _ = torch.min(weights_prd, dim=-1)
				mean_min_weight_value = torch.mean(min_weight_values)
			else:
				values, indices = torch.topk(weights_prd,k=self.top_k,dim=-1)
				min_weight_values, _ = torch.min(values, dim=-1)
				mean_min_weight_value = torch.mean(min_weight_values)
				masking_advantage = torch.sum(F.one_hot(indices, num_classes=self.num_agents), dim=-2)
				advantage = torch.sum(self.calculate_advantages(target_values, V_values, rewards, dones) * torch.transpose(masking_advantage,-1,-2),dim=-2)
		elif "greedy" in self.experiment_type:
			advantage = torch.sum(self.calculate_advantages(target_values, V_values, rewards, dones) * self.greedy_policy ,dim=-2)
		elif "relevant_set" in self.experiment_type:
			advantage = torch.sum(self.calculate_advantages(target_values, V_values, rewards, dones) * self.relevant_set ,dim=-2)

		if "scaled" in self.experiment_type and episode > self.steps_to_take:
			if "prd_soft_adv" in self.experiment_type:
				advantage = advantage*self.num_agents
			elif "top" in self.experiment_type:
				advantage = advantage*(self.num_agents/self.top_k)

		return advantage, masking_advantage

	def calculate_policy_loss(self, probs, actions, entropy, advantage):
		probs = Categorical(probs)
		policy_loss = -probs.log_prob(actions) * advantage.detach()
		policy_loss = policy_loss.mean() - self.entropy_pen*entropy

		return policy_loss

	def update_parameters(self):
		self.episode += 1

		if self.select_above_threshold > self.threshold_min and "prd_above_threshold_decay" in self.experiment_type:
			self.select_above_threshold = self.select_above_threshold - self.threshold_delta

		if self.threshold_max >= self.select_above_threshold and "prd_above_threshold_ascend" in self.experiment_type:
			self.select_above_threshold = self.select_above_threshold + self.threshold_delta

		# annealing entropy pen
		if self.entropy_pen > 0:
			self.entropy_pen = self.entropy_pen - self.entropy_delta


	def update(self,states_critic,one_hot_actions,actions,states_actor,rewards,dones,episode):

		'''
		Getting the probability mass function over the action space for each agent
		'''
		# start_forward_time = time.process_time()

		probs = self.policy_network.forward(states_actor)

		'''
		Calculate V values
		'''
		V_values, weights_value = self.critic_network.forward(states_critic, probs.detach(), one_hot_actions)
		V_values = V_values.reshape(-1,self.num_agents,self.num_agents)

		if self.critic_loss_type == "MC":
			discounted_rewards = self.calculate_returns(rewards,self.gamma).unsqueeze(-2).repeat(1,self.num_agents,1).to(self.device)
			target_V_values = torch.transpose(discounted_rewards,-1,-2)
			old_V_values = target_V_values
		elif self.critic_loss_type == "TD_lambda":
			with torch.no_grad():
				old_V_values, _ = self.target_critic_network.forward(states_critic, probs.detach(), one_hot_actions)
			old_V_values = old_V_values.reshape(self.update_after_episodes, -1, self.num_agents, self.num_agents)
			target_V_values = self.build_td_lambda_targets(rewards.reshape(self.update_after_episodes, -1, self.num_agents).unsqueeze(-1), dones.reshape(self.update_after_episodes, -1, self.num_agents).unsqueeze(-1), old_V_values).reshape(-1, self.num_agents, self.num_agents)
			old_V_values = old_V_values.reshape(-1, self.num_agents, self.num_agents)

		# end_forward_time = time.process_time()
		# self.forward_time += end_forward_time - start_forward_time

		# target_V_values = V_values.clone()

		if "prd" in self.experiment_type:
			weights_prd = weights_value
		else:
			weights_prd = None
	

		value_loss = self.calculate_value_loss(V_values, target_V_values, weights_value)
	
		# policy entropy
		entropy = -torch.mean(torch.sum(probs * torch.log(torch.clamp(probs, 1e-10,1.0)), dim=2))

		advantage, masking_advantage = self.calculate_advantages_based_on_exp(old_V_values, V_values, rewards, dones, weights_prd, episode)

		if "prd_avg" in self.experiment_type:
			agent_groups_over_episode = torch.sum(masking_advantage,dim=0)
			avg_agent_group_over_episode = torch.mean(agent_groups_over_episode.float())
		elif "threshold" in self.experiment_type:
			agent_groups_over_episode = torch.sum(torch.sum(masking_advantage.float(), dim=-2),dim=0)/masking_advantage.shape[0]
			avg_agent_group_over_episode = torch.mean(agent_groups_over_episode)
	
		policy_loss = self.calculate_policy_loss(probs, actions, entropy, advantage)
		# # ***********************************************************************************
		
		# start_update_time = time.process_time()

		# **********************************
		self.critic_optimizer.zero_grad()
		value_loss.backward(retain_graph=False)
		grad_norm_value = torch.nn.utils.clip_grad_norm_(self.critic_network.parameters(),self.grad_clip_critic)
		self.critic_optimizer.step()


		self.policy_optimizer.zero_grad()
		policy_loss.backward(retain_graph=False)
		grad_norm_policy = torch.nn.utils.clip_grad_norm_(self.policy_network.parameters(),self.grad_clip_actor)
		self.policy_optimizer.step()

		# end_update_time = time.process_time()
		# self.update_time += end_update_time - start_update_time


		self.update_parameters()

		# if "prd" in self.experiment_type and self.env_name in ["paired_by_sharing_goals", "crossing_partially_coop", "crossing_team_greedy"]:
		# 	relevant_set_error_rate = torch.mean(masking_advantage*self.relevant_set)
		# else:
		# 	relevant_set_error_rate = -1

		if episode % self.network_update_interval == 0:
			# Copy new weights into old critic
			self.target_critic_network.load_state_dict(self.critic_network.state_dict())


		self.plotting_dict = {
		"value_loss": value_loss,
		"policy_loss": policy_loss,
		"entropy": entropy,
		"grad_norm_value":grad_norm_value,
		"grad_norm_policy": grad_norm_policy,
		"weights_value": weights_value,
		# "relevant_set_error_rate":relevant_set_error_rate,
		}

		if "threshold" in self.experiment_type:
			self.plotting_dict["agent_groups_over_episode"] = agent_groups_over_episode
			self.plotting_dict["avg_agent_group_over_episode"] = avg_agent_group_over_episode
		if "prd_top" in self.experiment_type:
			self.plotting_dict["mean_min_weight_value"] = mean_min_weight_value

		if self.comet_ml is not None:
			self.plot(episode)
