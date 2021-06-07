import numpy as np
import torch 
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
from torch.autograd import Variable
from torch.distributions import Categorical
from a2c import ScalarDotProductCriticNetwork, ScalarDotProductPolicyNetwork, ScalarDotProductCriticNetworkDualAttention, ScalarDotProductPolicyNetworkDualAttention
import torch.nn.functional as F

class A2CAgent:

	def __init__(
		self, 
		env, 
		dictionary
		):
		self.env_name = dictionary["env"]
		self.env = env
		self.value_lr = dictionary["value_lr"]
		self.policy_lr = dictionary["policy_lr"]
		self.gamma = dictionary["gamma"]
		self.trace_decay = dictionary["trace_decay"]
		self.top_k = dictionary["top_k"]
		self.l1_pen = dictionary["l1_pen"]
		# Used for masking advantages above a threshold
		self.select_above_threshold = dictionary["select_above_threshold"]
		# cut the tail of softmax --> Used in softmax with normalization
		self.softmax_cut_threshold = dictionary["softmax_cut_threshold"]

		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		# self.device = "cpu"
		
		self.num_agents = self.env.n
		self.num_actions = self.env.action_space[0].n

		# CROWD NAV
		if self.env_name == "crowd_nav":
			self.num_agents = dictionary["num_agents"]
			self.num_people = dictionary["num_people"]

		self.gif = dictionary["gif"]
		self.gae = dictionary["gae"]
		self.norm_adv = dictionary["norm_adv"]
		self.norm_rew = dictionary["norm_rew"]

		self.experiment_type = dictionary["experiment_type"]

		self.anneal_entropy_pen = dictionary["anneal_entropy_pen"]
		if self.anneal_entropy_pen:
			self.steps_done = 0
			self.entropy_pen_start = dictionary["entropy_pen"]
			self.entropy_pen_end = dictionary["entropy_pen_end"]
			self.entropy_pen_decay = dictionary["entropy_pen_decay"]
			self.entropy_pen = self.entropy_pen_end + (self.entropy_pen_start-self.entropy_pen_end)*math.exp(-1*self.steps_done / self.entropy_pen_decay)
		else:
			self.entropy_pen = dictionary["entropy_pen"]


		self.greedy_policy = torch.zeros(self.num_agents,self.num_agents).to(self.device)
		for i in range(self.num_agents):
			self.greedy_policy[i][i] = 1

		print(self.experiment_type)

		# TD LAMBDA
		self.lambda_ = dictionary["td_lambda"]

		# Loss type
		self.critic_loss_type = dictionary["critic_loss_type"]


		# SCALAR DOT PRODUCT
		if self.env_name in ["collision_avoidance", "multi_circular"]:
			self.obs_input_dim = 2*3
			self.obs_output_dim = 16
			self.obs_act_input_dim = self.obs_input_dim + self.num_actions # (pose,vel,goal pose, paired agent goal pose) --> observations 
			self.obs_act_output_dim = 16
			self.final_input_dim = self.obs_act_output_dim #+ self.obs_input_dim #self.obs_z_output_dim + self.weight_input_dim
			self.final_output_dim = 1
			self.critic_network = ScalarDotProductCriticNetwork(self.obs_input_dim, self.obs_output_dim, self.obs_act_input_dim, self.obs_act_output_dim, self.final_input_dim, self.final_output_dim, self.num_agents, self.num_actions, self.softmax_cut_threshold).to(self.device)

			self.obs_input_dim = 2*3
			self.obs_output_dim = 16
			self.final_input_dim = self.obs_output_dim
			self.final_output_dim = self.num_actions
			self.policy_network = ScalarDotProductPolicyNetwork(self.obs_input_dim, self.obs_output_dim, self.final_input_dim, self.final_output_dim, self.num_agents, self.num_actions, self.softmax_cut_threshold).to(self.device)

		elif self.env_name == "paired_by_sharing_goals":
			self.obs_input_dim = 2*4
			self.obs_output_dim = 64
			self.obs_act_input_dim = self.obs_input_dim + self.num_actions # (pose,vel,goal pose, paired agent goal pose) --> observations 
			self.obs_act_output_dim = 64
			self.final_input_dim = self.obs_act_output_dim
			self.final_output_dim = 1
			self.critic_network = ScalarDotProductCriticNetwork(self.obs_input_dim, self.obs_output_dim, self.obs_act_input_dim, self.obs_act_output_dim, self.final_input_dim, self.final_output_dim, self.num_agents, self.num_actions, self.softmax_cut_threshold).to(self.device)

			self.obs_input_dim = 2*3
			self.obs_output_dim = 16
			self.final_input_dim = self.obs_output_dim
			self.final_output_dim = self.num_actions
			self.policy_network = ScalarDotProductPolicyNetwork(self.obs_input_dim, self.obs_output_dim, self.final_input_dim, self.final_output_dim, self.num_agents, self.num_actions, self.softmax_cut_threshold).to(self.device)

		elif self.env_name == "crowd_nav":
			self.obs_agent_input_dim = 2*3
			self.obs_agent_output_dim = 64
			self.obs_act_agent_input_dim = self.obs_agent_input_dim + self.num_actions # (pose,vel,goal pose, paired agent goal pose) --> observations 
			self.obs_act_agent_output_dim = 64
			self.obs_people_input_dim = 2*2
			self.obs_people_output_dim = 64
			self.obs_act_people_input_dim = self.obs_people_input_dim + self.num_actions # (pose,vel,goal pose, paired agent goal pose) --> observations 
			self.obs_act_people_output_dim = 64
			self.final_input_dim = self.obs_act_agent_output_dim + self.obs_act_people_output_dim
			self.final_output_dim = 1
			self.critic_network = ScalarDotProductCriticNetworkDualAttention(self.obs_agent_input_dim, self.obs_agent_output_dim, self.obs_act_agent_input_dim, self.obs_act_agent_output_dim, self.obs_people_input_dim, self.obs_people_output_dim, self.obs_act_people_input_dim, self.obs_act_people_output_dim, self.final_input_dim, self.final_output_dim, self.num_agents, self.num_people, self.num_actions, self.softmax_cut_threshold).to(self.device)
			
			self.obs_agent_input_dim = 2*3
			self.obs_agent_output_dim = 64
			self.obs_people_input_dim = 2*2 # (pose,vel,goal pose, paired agent goal pose) --> observations 
			self.obs_people_output_dim = 64
			self.final_input_dim = self.obs_agent_output_dim + self.obs_people_output_dim
			self.final_output_dim = self.num_actions
			self.policy_network = ScalarDotProductPolicyNetworkDualAttention(self.obs_agent_input_dim, self.obs_agent_output_dim, self.obs_people_input_dim, self.obs_people_output_dim, self.final_input_dim, self.final_output_dim, self.num_agents, self.num_people, self.num_actions, self.softmax_cut_threshold).to(self.device)

		
		if self.critic_loss_type == "td_1":
			self.critic_network_target = ScalarDotProductCriticNetwork(self.obs_input_dim, self.obs_output_dim, self.obs_act_input_dim, self.obs_act_output_dim, self.final_input_dim, self.final_output_dim, self.num_agents, self.num_actions, self.softmax_cut_threshold).to(self.device)
			self.critic_network_target.load_state_dict(self.critic_network.state_dict())
			self.tau = dictionary["tau"]
			self.critic_update_type = dictionary["critic_update_type"]
		
		

		# Loading models
		if dictionary["load_models"]:
			if "cpu" in self.device:
				self.critic_network.load_state_dict(torch.load(dicitionary["critic_saved_path"],map_location=torch.device('cpu')))
				self.policy_network.load_state_dict(torch.load(dicitionary["actor_saved_path"],map_location=torch.device('cpu')))
			else:
				# For GPU
				self.critic_network.load_state_dict(torch.load(dicitionary["critic_saved_path"]))
				self.policy_network.load_state_dict(torch.load(dicitionary["actor_saved_path"]))


		self.critic_optimizer = optim.Adam(self.critic_network.parameters(),lr=self.value_lr)
		self.policy_optimizer = optim.Adam(self.policy_network.parameters(),lr=self.policy_lr)


	def get_action(self,state):
		state = torch.FloatTensor([state]).to(self.device)
		dists, _ = self.policy_network.forward(state)
		index = [Categorical(dist).sample().cpu().detach().item() for dist in dists[0]]
		return index


	def soft_update(self, local_model, target_model):
		for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
			target_param.data.copy_(self.tau*local_param.data + (1.0-self.tau)*target_param.data)


	def hard_update(self,local_model, target_model):
		target_model.load_state_dict(local_model.state_dict())



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
		
		

	def calculate_deltas(self, values, rewards, dones):
		target_values = []
		next_value = 0
		rewards = rewards.unsqueeze(-1)
		dones = dones.unsqueeze(-1)
		masks = 1-dones
		for t in reversed(range(0, len(rewards))):
			value_target = rewards[t] + (self.gamma * next_value * masks[t])
			next_value = values.data[t]
			target_values.insert(0,value_target)
		target_values = torch.stack(target_values)

		return target_values


	def nstep_returns(self,values, rewards, dones):
		target_values = self.calculate_deltas(values, rewards, dones)
		target_Vs = self.calculate_returns(target_values, self.gamma*self.lambda_)
		return target_Vs




	def update(self,states_critic,next_states_critic,one_hot_actions,one_hot_next_actions,actions,states_actor,next_states_actor,rewards,dones,states_critic_people=None,next_states_critic_people=None,one_hot_actions_people=None,one_hot_next_actions_people=None,states_actor_people=None,next_states_actor_people=None):

		if self.env_name in ["paired_by_sharing_goals", "collision_avoidance", "multi_circular"]:
			probs, weight_policy = self.policy_network.forward(states_actor)

			V_values, weights = self.critic_network.forward(states_critic, probs.detach(), one_hot_actions)
			V_values = V_values.reshape(-1,self.num_agents,self.num_agents)
		elif self.env_name in ["crowd_nav"]:
			probs, weight_policy, weight_policy_people = self.policy_network.forward(states_actor,states_actor_people)

			V_values, weights, weights_people = self.critic_network.forward(states_critic, probs.detach(), one_hot_actions, states_critic_people, one_hot_actions_people)
			V_values = V_values.reshape(-1,self.num_agents,self.num_agents)

		
	# # ***********************************************************************************
	# 	#update critic (value_net)
		# we need a TxNxN vector so inflate the discounted rewards by N --> cloning the discounted rewards for an agent N times
		discounted_rewards = self.calculate_returns(rewards,self.gamma).unsqueeze(-2).repeat(1,self.num_agents,1).to(self.device)
		discounted_rewards = torch.transpose(discounted_rewards,-1,-2)
		if self.critic_loss_type == "monte_carlo":
			value_loss = F.smooth_l1_loss(V_values,discounted_rewards)
			
		elif self.critic_loss_type == "td_1":
			if self.env_name in ["paired_by_sharing_goals", "collision_avoidance", "multi_circular"]:
				next_probs, _ = self.policy_network.forward(next_states_actor)
				V_values_next, _ = self.critic_network_target.forward(next_states_critic, next_probs.detach(), one_hot_next_actions)
			elif self.env_name in ["crowd_nav"]:
				next_probs, _, _ = self.policy_network.forward(next_states_actor, next_states_actor_people)
				V_values_next, _, _ = self.critic_network_target.forward(next_states_critic, next_probs.detach(), one_hot_next_actions, next_states_critic_people, one_hot_actions_people)

			V_values_next = V_values_next.reshape(-1,self.num_agents,self.num_agents)
			V_values_target = rewards.unsqueeze(-1) + self.gamma*V_values_next*(1-dones.unsqueeze(-1))
			value_loss = F.smooth_l1_loss(V_values,V_values_target.detach())

			if self.critic_update_type == "soft":
				self.soft_update(self.critic_network, self.critic_network_target)
			elif self.critic_update_type == "hard":
				self.hard_update(self.critic_network, self.critic_network_target)

		elif self.critic_loss_type == "td_lambda":
			V_values_target = self.nstep_returns(V_values, rewards, dones)
			value_loss = F.smooth_l1_loss(V_values,V_values_target.detach())

		if "prd" in self.experiment_type:
			weights_off_diagonal = weights * (1 - torch.eye(self.num_agents,device=self.device))
			l1_weights = torch.mean(weights_off_diagonal)
			value_loss += self.l1_pen*l1_weights
		
		# # ***********************************************************************************
	# 	#update actor (policy net)
	# # ***********************************************************************************
		entropy = -torch.mean(torch.sum(probs * torch.log(torch.clamp(probs, 1e-10,1.0)), dim=2))

		# summing across each agent j to get the advantage
		# so we sum across the second last dimension which does A[t,j] = sum(V[t,i,j] - discounted_rewards[t,i])
		advantage = None
		if self.experiment_type == "without_prd":
			advantage = torch.sum(self.calculate_advantages(discounted_rewards, V_values, rewards, dones),dim=-2)
		elif "top" in self.experiment_type:
			values, indices = torch.topk(weights,k=self.top_k,dim=-1)
			masking_advantage = torch.transpose(torch.sum(F.one_hot(indices, num_classes=self.num_agents), dim=-2),-1,-2)
			advantage = torch.sum(self.calculate_advantages(discounted_rewards, V_values, rewards, dones) * masking_advantage,dim=-2)
		elif self.experiment_type in "above_threshold":
			masking_advantage = torch.transpose((weights>self.select_above_threshold).int(),-1,-2)
			advantage = torch.sum(self.calculate_advantages(discounted_rewards, V_values, rewards, dones) * masking_advantage,dim=-2)
		elif self.experiment_type == "with_prd_soft_adv":
			advantage = torch.sum(self.calculate_advantages(discounted_rewards, V_values, rewards, dones) * weights.transpose(-1,-2) ,dim=-2)
		elif self.experiment_type == "greedy_policy":
			advantage = torch.sum(self.calculate_advantages(discounted_rewards, V_values, rewards, dones) * self.greedy_policy ,dim=-2)


		probs = Categorical(probs)
		policy_loss = -probs.log_prob(actions) * advantage.detach()
		policy_loss = policy_loss.mean() - self.entropy_pen*entropy
	# # ***********************************************************************************
		
	# **********************************
		self.critic_optimizer.zero_grad()
		value_loss.backward(retain_graph=False)
		grad_norm_value = torch.nn.utils.clip_grad_norm_(self.critic_network.parameters(),0.5)
		self.critic_optimizer.step()


		self.policy_optimizer.zero_grad()
		policy_loss.backward(retain_graph=False)
		grad_norm_policy = torch.nn.utils.clip_grad_norm_(self.policy_network.parameters(),0.5)
		self.policy_optimizer.step()

		if self.anneal_entropy_pen:
			self.steps_done+=1
			self.entropy_pen = self.entropy_pen_end + (self.entropy_pen_start-self.entropy_pen_end)*math.exp(-1*self.steps_done / self.entropy_pen_decay)

		# V values
		if self.env_name in ["paired_by_sharing_goals", "collision_avoidance", "multi_circular"]:
			return value_loss,policy_loss,entropy,grad_norm_value,grad_norm_policy,weights, weight_policy
		elif self.env_name in ["crowd_nav"]:
			return value_loss,policy_loss,entropy,grad_norm_value,grad_norm_policy,weights, weight_policy, weights_people, weight_policy_people