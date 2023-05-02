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
		
		self.num_agents = self.env.num_agents
		self.num_actions = self.env.action_space("pursuer_0").n

		self.epsilon_start = dictionary["epsilon_start"]
		self.epsilon = self.epsilon_start
		self.epsilon_end = dictionary["epsilon_end"]
		self.epsilon_episode_steps = dictionary["epsilon_episode_steps"]
		self.episode = 0
		self.target_critic_update = dictionary["target_critic_update"]

		self.grad_clip_critic = dictionary["grad_clip_critic"]
		self.grad_clip_actor = dictionary["grad_clip_actor"]

		self.height = dictionary["height"]
		self.width = dictionary["width"]
		self.channel = dictionary["channel"]

		
		self.critic_network = TransformerCritic(self.height, self.width, self.channel, self.num_agents, self.num_actions, self.device).to(self.device)
		self.target_critic_network = TransformerCritic(self.height, self.width, self.channel, self.num_agents, self.num_actions, self.device).to(self.device)

		self.target_critic_network.load_state_dict(self.critic_network.state_dict())

		# MLP POLICY
		self.policy_network = MLPPolicy(self.height, self.width, self.channel, self.num_agents, self.num_actions).to(self.device)

		if dictionary["load_models"]:
			# Loading models
			if torch.cuda.is_available() is False:
				# For CPU
				self.critic_network.load_state_dict(torch.load(dictionary["model_path_value"],map_location=torch.device('cpu')))
				self.target_critic_network.load_state_dict(torch.load(dictionary["model_path_value"],map_location=torch.device('cpu')))
				self.policy_network.load_state_dict(torch.load(dictionary["model_path_policy"],map_location=torch.device('cpu')))
			else:
				# For GPU
				self.critic_network.load_state_dict(torch.load(dictionary["model_path_value"]))
				self.target_critic_network.load_state_dict(torch.load(dictionary["model_path_value"]))
				self.policy_network.load_state_dict(torch.load(dictionary["model_path_policy"]))

		
		self.critic_optimizer = optim.Adam(self.critic_network.parameters(),lr=self.value_lr)
		self.policy_optimizer = optim.Adam(self.policy_network.parameters(),lr=self.policy_lr)

		self.comet_ml = None
		if dictionary["save_comet_ml_plot"]:
			self.comet_ml = comet_ml


	def get_action(self, state_policy, greedy=False):
		if np.random.uniform() < self.epsilon:
			actions = [np.random.choice(self.num_actions) for _ in range(self.num_agents)]
		else:
			with torch.no_grad():
				state_policy = torch.FloatTensor(state_policy).permute(0, 3, 1, 2).to(self.device)
				dists = self.policy_network(state_policy).squeeze(0)
				if greedy:
					actions = [dist.argmax().detach().cpu().item() for dist in dists]
				else:
					actions = [Categorical(dist).sample().detach().cpu().item() for dist in dists]

		return actions


	def calculate_advantages(self, Q_values, baseline):
		advantages = Q_values - baseline
		
		if self.norm_adv:
			advantages = (advantages - advantages.mean()) / advantages.std()
		
		return advantages


	# def calculate_deltas(self, values, rewards, dones):
	# 	deltas = []
	# 	next_value = 0
	# 	rewards = rewards
	# 	dones = dones
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

	def build_td_lambda_targets(self, rewards, terminated, target_qs):
		# Assumes  <target_qs > in B*T*A and <reward >, <terminated >  in B*T*A, <mask > in (at least) B*T-1*1
		# Initialise  last  lambda -return  for  not  terminated  episodes
		ret = target_qs.new_zeros(*target_qs.shape)
		ret = target_qs * (1-terminated)
		# ret[:, -1] = target_qs[:, -1] * (1 - (torch.sum(terminated, dim=1)>0).int())
		# Backwards  recursive  update  of the "forward  view"
		for t in range(ret.shape[1] - 2, -1,  -1):
			ret[:, t] = self.lambda_ * self.gamma * ret[:, t + 1] + \
						(rewards[:, t] + (1 - self.lambda_) * self.gamma * target_qs[:, t + 1] * (1 - terminated[:, t]))
		# Returns lambda-return from t=0 to t=T-1, i.e. in B*T-1*A
		# return ret[:, 0:-1]
		return ret


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
		self.comet_ml.log_metric('Value_Loss',self.plotting_dict["value_loss"],episode)
		self.comet_ml.log_metric('Grad_Norm_Value',self.plotting_dict["grad_norm_value"],episode)
		self.comet_ml.log_metric('Policy_Loss',self.plotting_dict["policy_loss"],episode)
		self.comet_ml.log_metric('Grad_Norm_Policy',self.plotting_dict["grad_norm_policy"],episode)
		self.comet_ml.log_metric('Entropy',self.plotting_dict["entropy"],episode)

		entropy_weights = -torch.mean(torch.sum(self.plotting_dict["weights_value"]* torch.log(torch.clamp(self.plotting_dict["weights_value"], 1e-10,1.0)), dim=2))
		self.comet_ml.log_metric('Critic_Weight_Entropy', entropy_weights.item(), episode)

		
	def calculate_value_loss(self, Q_values, Q_values_target, weights):
		value_loss = F.smooth_l1_loss(Q_values, Q_values_target)

		if self.critic_entropy_pen != 0:
			weight_entropy = -torch.mean(torch.sum(weights * torch.log(torch.clamp(weights, 1e-10,1.0)), dim=2))
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


	def update(self, states, one_hot_actions, actions, rewards, dones, episode):		
		'''
		Getting the probability mass function over the action space for each agent
		'''
		probs = self.policy_network(states)

		Q_values, weights_value = self.critic_network(states, one_hot_actions)
		Q_values_act_chosen = torch.sum(Q_values.reshape(-1,self.num_agents, self.num_actions) * one_hot_actions, dim=-1)
		V_values_baseline = torch.sum(Q_values.reshape(-1,self.num_agents, self.num_actions) * probs.detach(), dim=-1)
	
		target_Q_values, _ = self.target_critic_network(states, one_hot_actions)
		target_Q_values_act_chosen = torch.sum(target_Q_values.reshape(-1,self.num_agents, self.num_actions) * one_hot_actions, dim=-1)

		# Q_values_target = self.nstep_returns(target_Q_values, rewards, dones).detach()
		Q_values_target = self.build_td_lambda_targets(rewards, dones, target_Q_values_act_chosen)

		value_loss = self.calculate_value_loss(Q_values_act_chosen, Q_values_target, weights_value)

		advantage = self.calculate_advantages(Q_values_act_chosen, V_values_baseline)

		entropy = -torch.mean(torch.sum(probs * torch.log(torch.clamp(probs, 1e-10,1.0)), dim=-1))
	
		policy_loss = self.calculate_policy_loss(probs, actions, advantage)
		# # ***********************************************************************************
			
		
		self.critic_optimizer.zero_grad()
		value_loss.backward(retain_graph=False)
		grad_norm_value = torch.nn.utils.clip_grad_norm_(self.critic_network.parameters(),self.grad_clip_critic)
		self.critic_optimizer.step()


		self.policy_optimizer.zero_grad()
		policy_loss.backward(retain_graph=False)
		grad_norm_policy = torch.nn.utils.clip_grad_norm_(self.policy_network.parameters(),self.grad_clip_actor)
		self.policy_optimizer.step()

		self.update_parameters()

		self.plotting_dict = {
		"value_loss": value_loss.item(),
		"policy_loss": policy_loss.item(),
		"entropy": entropy.item(),
		"grad_norm_value":grad_norm_value,
		"grad_norm_policy": grad_norm_policy,
		"weights_value": weights_value.detach(),
		}

		if self.comet_ml is not None:
			self.plot(episode)