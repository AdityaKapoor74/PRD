import numpy as np
import torch 
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
from torch.autograd import Variable
from torch.distributions import Categorical
from a2c import *
import torch.nn.functional as F
from fraction import Fraction

class A2CAgent:

	def __init__(self,env,value_lr=2e-4, policy_lr=2e-4, actorcritic_lr=2e-4, entropy_pen=0.008, gamma=0.99, lambda_=0.01):
		self.env = env
		self.value_lr = value_lr
		self.policy_lr = policy_lr
		self.gamma = gamma
		self.entropy_pen = entropy_pen
		self.lambda_ = lambda_

		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		
		self.num_agents = self.env.n

		# Agent 1 is paired with Agent 4 and Agent 2 is paired with Agent 3
		self.dict = {}
		self.dict_fraction = {}
		self.accuracy = {'1':{},'2':{},'3':{},'4':{}}
		self.precision = {'1':{},'2':{},'3':{},'4':{}}
		self.recall = {'1':{},'2':{},'3':{},'4':{}}

		# # Shared Network
		# self.lr = actorcritic_lr
		# self.input_dim = self.env.observation_space[0].shape[0]
		# self.value_output_dim = 1
		# self.policy_output_dim = self.action_space[0].n
		# self.actorcritic = CentralizedActorCritic(self.input_dim,self.value_output_dim,self.policy_output_dim)
		# self.optimizer = optim.Adam(self.actorcritic.parameters(),lr=self.lr)
		# # Loading models
		# model_path = "./models/shared_network/3_agents/value_net_lr_2e-4_policy_net_lr_2e-4_with_grad_norm_0.5_entropy_pen_0.008_xavier_uniform_init_clamp_logs.pt"
		# self.actorcritic.load_state_dict(torch.load(model_path_value,map_location=torch.device('cpu'))) # for CPU
		# self.actorcritic.load_state_dict(torch.load(model_path_value)) # for GPU 



		# # Two Headed Network
		# self.lr = actorcritic_lr
		# self.value_input_dim = self.env.observation_space[0].shape[0]
		# self.policy_input_dim = self.env.observation_space[0].shape[0]
		# self.value_output_dim = 1
		# self.policy_output_dim = self.action_space[0].n
		# self.actorcritic = TwoHeadedActorCritic(self.value_input_dim,self.policy_input_dim,self.value_output_dim,self.policy_output_dim)
		# self.optimizer = optim.Adam(self.actorcritic.parameters(),lr=self.lr)
		# # Loading models
		# model_path = "./models/two_headed_network/3_agents/value_net_lr_2e-4_policy_net_lr_2e-4_with_grad_norm_0.5_entropy_pen_0.008_xavier_uniform_init_clamp_logs.pt"
		# self.actorcritic.load_state_dict(torch.load(model_path_value,map_location=torch.device('cpu'))) # for CPU
		# self.actorcritic.load_state_dict(torch.load(model_path_value)) # for GPU 


# 		# Separate Networks
		# self.value_input_dim = 2*3*self.num_agents
		# self.value_output_dim = 1
		# self.policy_input_dim = 2*(3+self.num_agents-1)
		# self.policy_output_dim = self.env.action_space[0].n
		# self.value_network = ValueNetwork(self.value_input_dim,self.value_output_dim).to(self.device)
		# self.policy_network = PolicyNetwork(self.policy_input_dim,self.policy_output_dim).to(self.device)
		# self.value_optimizer = optim.Adam(self.value_network.parameters(),lr=self.value_lr)
		# self.policy_optimizer = optim.Adam(self.policy_network.parameters(),lr=self.policy_lr)
# 		# Loading models
# 		model_path_value = "./models/separate_net/4_agents/value_net_lr_2e-4_policy_lr_2e-4_with_grad_norm_0.5_entropy_pen_0.008_xavier_uniform_init_clamp_logs.pt"
# 		model_path_policy = "./models/separate_net/4_agents/policy_net_lr_2e-4_value_lr_2e-4_with_grad_norm_0.5_entropy_pen_0.008_xavier_uniform_init_clamp_logs.pt"
		# # For CPU
		# self.value_network.load_state_dict(torch.load(model_path_value,map_location=torch.device('cpu')))
		# self.policy_network.load_state_dict(torch.load(model_path_policy,map_location=torch.device('cpu')))
		# # For GPU
# 		self.value_network.load_state_dict(torch.load(model_path_value))
# 		self.policy_network.load_state_dict(torch.load(model_path_policy))


		# Separate Network with action conditioning
		self.value_input_dim = 2*3*self.num_agents # for pose,vel and landmark for every agent
		self.value_output_dim = 1
		self.weight_output_dim = 2*(self.num_agents-1)
		self.num_actions = self.env.action_space[0].n
		self.policy_input_dim = 2*(3+self.num_agents-1) #2 for pose, 2 for vel and 2 for goal of current agent and rest for relative position of other agents 2 each
		self.policy_output_dim = self.env.action_space[0].n
		self.value_network = ValueNetwork_(self.value_input_dim,self.num_agents,self.num_actions,self.weight_output_dim,self.value_output_dim).to(self.device)
		self.policy_network = PolicyNetwork_(self.policy_input_dim,self.policy_output_dim).to(self.device)
		self.value_optimizer = optim.Adam(self.value_network.parameters(),lr=self.value_lr)
		self.policy_optimizer = optim.Adam(self.policy_network.parameters(),lr=self.policy_lr)
		# Loading models
		# model_path_value = "./models/separate_net_with_action_conditioning/4_agents/value_net_lr_2e-4_policy_lr_2e-4_with_grad_norm_0.5_entropy_pen_0.008_xavier_uniform_init_clamp_logs_lambda_1.0.pt"
		# model_path_policy = "./models/separate_net_with_action_conditioning/4_agents/policy_net_lr_2e-4_value_lr_2e-4_with_grad_norm_0.5_entropy_pen_0.008_xavier_uniform_init_clamp_logs_lambda_1.0.pt"
		# For CPU
		# self.value_network.load_state_dict(torch.load(model_path_value,map_location=torch.device('cpu')))
		# self.policy_network.load_state_dict(torch.load(model_path_policy,map_location=torch.device('cpu')))
# 		# # For GPU
		# self.value_network.load_state_dict(torch.load(model_path_value))
		# self.policy_network.load_state_dict(torch.load(model_path_policy))


		
	

	def get_action(self,state):
		state = torch.FloatTensor(state).to(self.device)
		# _,dists = self.actorcritic.forward(state) # for shared network and two headed network
		dists = self.policy_network.forward(state) # for separate network
		probs = Categorical(dists)
		index = probs.sample().cpu().detach().item()

		return index



	def get_one_hot_encoding(self,actions):
		one_hot = torch.zeros([actions.shape[0], self.num_agents, self.env.action_space[0].n], dtype=torch.int32)
		for i in range(one_hot.shape[0]):
			for j in range(self.num_agents):
				one_hot[i][j][int(actions[i][j].item())] = 1

		return one_hot



	def calculate_advantages(self,returns, values, normalize = False):
	
		advantages = returns - values
		
		if normalize:
			
			advantages = (advantages - advantages.mean()) / advantages.std()
			
		return advantages


	def calculate_returns(self,rewards, discount_factor, normalize = False):
	
		returns = []
		R = 0
		
		for r in reversed(rewards):
			R = r + R * discount_factor
			returns.insert(0, R)
		
		returns_tensor = torch.stack(returns)
		
		if normalize:
			
			returns_tensor = (returns_tensor - returns_tensor.mean()) / returns_tensor.std()
			
		return returns_tensor

	def calculate_frequency_accuracy_precision_recall(self,threshold,tensor_list):

		TP = [0]*self.num_agents
		FP = [0]*self.num_agents
		TN = [0]*self.num_agents
		FN = [0]*self.num_agents

		if threshold not in self.dict:
			self.dict[threshold] = 0
			self.dict_fraction[threshold] = str(0)

		for i in range(tensor_list.shape[0]):
			for j in range(tensor_list.shape[1]):
				for k in range(tensor_list.shape[2]):

					if tensor_list[i][j][k]<=threshold:
						self.dict[threshold]+=1

					if j+1==1:
						if k==2:
							if tensor_list[i][j][k]>=threshold:
								TP[j]+=1
							else:
								FP[j]+=1
						else:
							if tensor_list[i][j][k]<threshold:
								TN[j]+=1
							else:
								FN[j]+=1

					elif j+1==2:
						if k==1:
							if tensor_list[i][j][k]>=threshold:
								TP[j]+=1
							else:
								FP[j]+=1
						else:
							if tensor_list[i][j][k]<threshold:
								TN[j]+=1
							else:
								FN[j]+=1

					elif j+1==3:
						if k==1:
							if tensor_list[i][j][k]>=threshold:
								TP[j]+=1
							else:
								FP[j]+=1
						else:
							if tensor_list[i][j][k]<threshold:
								TN[j]+=1
							else:
								FN[j]+=1

					elif j+1==4:
						if k==0:
							if tensor_list[i][j][k]>=threshold:
								TP[j]+=1
							else:
								FP[j]+=1
						else:
							if tensor_list[i][j][k]<threshold:
								TN[j]+=1
							else:
								FN[j]+=1

		for i in range(4):
			if (TP[i]+TN[i]+FP[i]+FN[i]) == 0:
				self.accuracy[str(i+1)][threshold] = 0
			else:
				self.accuracy[str(i+1)][threshold] = round((TP[i]+TN[i])/(TP[i]+TN[i]+FP[i]+FN[i]),4)
			if (TP[i]+FP[i]) == 0:
				self.precision[str(i+1)][threshold] = 0
			else:
				self.precision[str(i+1)][threshold] = round((TP[i]/(TP[i]+FP[i])),4)
			if (TP[i]+FN[i]) == 0:
				self.recall[str(i+1)][threshold] = 0
			else:
				self.recall[str(i+1)][threshold] = round((TP[i]/(TP[i]+FN[i])),4)

		self.dict_fraction[threshold] = str(Fraction(self.dict[threshold],self.dict[1]))


	def update(self,states_critic,states_actor,next_states_critic,next_states_actor,actions,rewards):

		'''
		Separate network with action conditioning
		'''
		probs = self.policy_network.forward(states_actor)
		one_hot_actions = self.get_one_hot_encoding(actions)


		weight_action,weight_prob,curr_Q = self.value_network.forward(states_critic,one_hot_actions,probs)

		for value in [1,1e-5,1e-4,1e-3,1e-2,1e-1]:
			self.calculate_frequency_accuracy_precision_recall(value,weight_action)


		with open('../../freq_accuracy_precision_recall_of_weight_actions'+str(self.lambda_)+'.txt', 'w+') as f:
			print("Fractions",file=f)
			print(self.dict_fraction, file=f)
			print("Frequencies", file=f)
			print(self.dict, file=f)
			print("Precision",file=f)
			print(self.precision, file=f)
			print("Recall",file=f)
			print(self.recall, file=f)
			print("Accuracy",file=f)
			print(self.accuracy, file=f)

		print("Frequencies of weight values")
		print(self.dict_fraction)
		print(self.dict)

		print("Precision")
		print(self.precision)

		print("Recall")
		print(self.recall)

		print("Accuracy")
		print(self.accuracy)



	# # ***********************************************************************************
	# 	#update critic (value_net)
		discounted_rewards = self.calculate_returns(rewards,self.gamma)

	# # ***********************************************************************************

	# # ***********************************************************************************
		value_targets = rewards.to(self.device) + torch.FloatTensor(discounted_rewards).to(self.device)
		value_targets = value_targets.unsqueeze(dim=-1)
		value_loss = F.smooth_l1_loss(curr_Q,value_targets,reduction='none')

		sum_weight_values = torch.sum(weight_action).to(self.device)

		loss = torch.FloatTensor([0]).to(self.device)

		for i in range(self.num_agents):
			loss += torch.sum(value_loss[:,i]) + self.lambda_*(sum_weight_values - torch.sum(weight_action[:,i]))

		

		value_loss = loss / (value_loss.shape[0]*value_loss.shape[1])

	# # ***********************************************************************************
	# 	#update actor (policy net)
	# # ***********************************************************************************

		entropy = -torch.mean(torch.sum(probs * torch.log(torch.clamp(probs, 1e-10,1.0)), dim=2))

		# summing across each agent 
		value_targets = torch.sum(value_targets,dim=1) 
		curr_Q = torch.sum(curr_Q,dim=1)


		advantage = value_targets - curr_Q
		probs = Categorical(probs)
		policy_loss = -probs.log_prob(actions) * advantage.detach()
		policy_loss = policy_loss.mean() - self.entropy_pen*entropy
	# # ***********************************************************************************
		
	# # ***********************************************************************************
		self.value_optimizer.zero_grad()
		value_loss.backward(retain_graph=False)
		grad_norm_value = torch.nn.utils.clip_grad_norm_(self.value_network.parameters(),0.5)
		self.value_optimizer.step()
		

		self.policy_optimizer.zero_grad()
		policy_loss.backward(retain_graph=False)
		grad_norm_policy = torch.nn.utils.clip_grad_norm_(self.policy_network.parameters(),0.5)
		self.policy_optimizer.step()
	# # ***********************************************************************************
		return value_loss,policy_loss,entropy,grad_norm_value,grad_norm_policy



		'''
		Separate networks
		'''
		
	# # ***********************************************************************************
	# 	#update critic (value_net)
	# 	curr_Q = self.value_network.forward(states_critic)

	# 	# discounted_rewards = np.asarray([[torch.sum(torch.FloatTensor([self.gamma**i for i in range(rewards[k][j:].size(0))])* rewards[k][j:]) for j in range(rewards.size(0))] for k in range(self.num_agents)])
	# 	# discounted_rewards = np.transpose(discounted_rewards)
	# 	discounted_rewards = self.calculate_returns(rewards,self.gamma)
	# # ***********************************************************************************

	# # ***********************************************************************************
	# 	value_targets = rewards.to(self.device) + torch.FloatTensor(discounted_rewards).to(self.device)
	# 	value_targets = value_targets.unsqueeze(dim=-1)
	# 	value_loss = F.smooth_l1_loss(curr_Q,value_targets)
	# # ***********************************************************************************

	# 	#update actor (policy net)
	# # ***********************************************************************************
	# 	probs = self.policy_network.forward(states_actor)

	# 	entropy = -torch.mean(torch.sum(probs * torch.log(torch.clamp(probs, 1e-10,1.0)), dim=2))

	# 	value_targets = torch.sum(value_targets,dim=1) 
	# 	curr_Q = torch.sum(curr_Q,dim=1)

	# 	advantage = value_targets - curr_Q
	# 	probs = Categorical(probs)
	# 	policy_loss = -probs.log_prob(actions) * advantage.detach()
	# 	policy_loss = policy_loss.mean() - self.entropy_pen*entropy
	# # ***********************************************************************************
		
	# # ***********************************************************************************
	# 	self.value_optimizer.zero_grad()
	# 	value_loss.backward(retain_graph=False)
	# 	grad_norm_value = torch.nn.utils.clip_grad_norm_(self.value_network.parameters(),0.5)
	# 	self.value_optimizer.step()
		

	# 	self.policy_optimizer.zero_grad()
	# 	policy_loss.backward(retain_graph=False)
	# 	grad_norm_policy = torch.nn.utils.clip_grad_norm_(self.policy_network.parameters(),0.5)
	# 	self.policy_optimizer.step()
	# # ***********************************************************************************

	# 	return value_loss,policy_loss,entropy,grad_norm_value,grad_norm_policy



	# 	'''
	# 	Shared networks
	# 	'''
		
	# # ***********************************************************************************
	# 	#update critic (value_net)
	# 	curr_Q,probs = self.value_network.forward(states)

	# # discounted_rewards = np.asarray([[torch.sum(torch.FloatTensor([self.gamma**i for i in range(rewards[k][j:].size(0))])* rewards[k][j:]) for j in range(rewards.size(0))] for k in range(self.num_agents)])
	# # discounted_rewards = np.transpose(discounted_rewards)
		# discounted_rewards = self.calculate_returns(rewards,self.gamma)
	# # ***********************************************************************************

	# # ***********************************************************************************
	# 	value_targets = rewards.to(self.device) + torch.FloatTensor(discounted_rewards).to(self.device)
	# 	value_targets = value_targets.unsqueeze(dim=-1)
	# 	value_loss = F.smooth_l1_loss(curr_Q,value_targets)
	# # ***********************************************************************************

	# 	#update actor (policy net)
	# # ***********************************************************************************

	# 	entropy = -torch.mean(torch.sum(probs * torch.log(torch.clamp(probs, 1e-10,1.0)), dim=2))

	# 	advantage = value_targets - curr_Q
	# 	probs = Categorical(probs)
	# 	policy_loss = -probs.log_prob(actions).unsqueeze(dim=-1) * advantage.detach()
	# 	policy_loss = policy_loss.mean() - self.entropy_pen*entropy
	# # ***********************************************************************************
		
	# # ***********************************************************************************
	# 	total_loss = value_loss+policy_loss
	# 	self.optimizer.zero_grad()
	# 	total_loss.backward(retain_graph=False)
	# 	grad_norm = torch.nn.utils.clip_grad_norm_(self.value_network.parameters(),0.5)
	# 	self.optimizer.step()
	# # ***********************************************************************************

	# 	return value_loss,policy_loss,entropy,grad_norm

