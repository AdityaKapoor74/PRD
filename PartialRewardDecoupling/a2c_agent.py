import numpy as np
import torch 
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
from torch.autograd import Variable
from torch.distributions import Categorical
from a2c import *
import torch.nn.functional as F
import gc

class A2CAgent:

	def __init__(self,env,value_lr=2e-4, policy_lr=2e-4, actorcritic_lr=2e-4, entropy_pen=0.008, gamma=0.99, lambda_=1):
		self.env = env
		self.value_lr = value_lr
		self.policy_lr = policy_lr
		self.gamma = gamma
		self.entropy_pen = entropy_pen
		self.lambda_ = lambda_

		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		
		self.num_agents = self.env.n
		self.dict = {}

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
# 		self.value_input_dim = self.env.observation_space[0].shape[0]
# 		self.value_output_dim = 1
# 		self.policy_input_dim = self.env.observation_space[0].shape[0]
# 		self.policy_output_dim = self.env.action_space[0].n
# 		self.value_network = ValueNetwork(self.value_input_dim,self.value_output_dim).to(self.device)
# 		self.policy_network = PolicyNetwork(self.policy_input_dim,self.policy_output_dim).to(self.device)
# 		self.value_optimizer = optim.Adam(self.value_network.parameters(),lr=self.value_lr)
# 		self.policy_optimizer = optim.Adam(self.policy_network.parameters(),lr=self.policy_lr)
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
		self.value_input_dim = self.env.observation_space[0].shape[0]
		self.value_output_dim = 1
		self.weight_output_dim = 2
		self.num_actions = self.env.action_space[0].n
		self.policy_input_dim = self.env.observation_space[0].shape[0]
		self.policy_output_dim = self.env.action_space[0].n
		self.value_network = ValueNetwork_(self.value_input_dim,self.num_agents,self.num_actions,self.weight_output_dim,self.value_output_dim).to(self.device)
		self.policy_network = PolicyNetwork_(self.policy_input_dim,self.policy_output_dim).to(self.device)
		self.value_optimizer = optim.Adam(self.value_network.parameters(),lr=self.value_lr)
		self.policy_optimizer = optim.Adam(self.policy_network.parameters(),lr=self.policy_lr)
		# Loading models
		# model_path_value = "./models/separate_net_with_action_conditioning/4_agents/value_net_lr_2e-4_policy_lr_2e-4_with_grad_norm_0.5_entropy_pen_0.008_xavier_uniform_init_clamp_logs.pt"
		# model_path_policy = "./models/separate_net_with_action_conditioning/4_agents/policy_net_lr_2e-4_value_lr_2e-4_with_grad_norm_0.5_entropy_pen_0.008_xavier_uniform_init_clamp_logs.pt"
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


	def calculate_frequency(self,value,tensor_list):

	 	if value not in self.dict:
	 		self.dict[value] = 0
	 	else:
	 		for num in tensor_list:
	 			if num<=value:
	 				self.dict[value]+=1


	def update(self,states,next_states,actions,rewards):

		'''
		Separate network with action conditioning
		'''
		weights = self.value_network.forward(states,None)
		weights = weights.permute(0,2,1)
		
		weight_prob = weights[0][0]
		weight_action = weights[0][1]

		for i in range(1,weights.shape[0]):
			weight_prob = torch.cat([weight_prob,weights[i][0]])
			weight_action = torch.cat([weight_action,weights[i][1]])

		

		for value in [1e-5,1e-4,1e-3,1e-2,1e-1,1]:
			self.calculate_frequency(value,weight_action)

		print("Frequencies of weight values")
		print(self.dict)


		weight_prob = weight_prob.reshape(-1,self.num_agents)
		weight_action = weight_action.reshape(-1,self.num_agents)

		weight_action_clone = weight_action.clone().unsqueeze(-1)

		probs = self.policy_network.forward(states)

		one_hot_actions = self.get_one_hot_encoding(actions)

		weight_prob = weight_prob.unsqueeze(-1)
		weight_prob = weight_prob.expand(weight_prob.shape[0],weight_prob.shape[1],self.num_actions)
		weight_action = weight_action.unsqueeze(-1)
		weight_action = weight_action.expand(weight_action.shape[0],weight_action.shape[1],self.num_actions)
		z = probs.cpu()*weight_prob.cpu()+one_hot_actions*weight_action.cpu()
		z = z.detach().numpy()

		# print(weight_action)
		# print(weight_action.squeeze(-1))

		states_value = []
		for k in range(states.shape[0]):
			for j in range(states.shape[1]): # states.shape[1]==self.num_agents
				z_copy = z[k].copy()
				z_copy = np.delete(z_copy,(j), axis=0)
				tmp = np.copy(states.cpu().numpy()[k][j])
				for i in range(self.num_agents-1):
					if i == self.num_agents-2:
						tmp = np.append(tmp,z_copy[i])
					else:
						tmp = np.insert(tmp,-(self.num_agents-2-i),z_copy[i])
				states_value.append(tmp)
		
		states_value = torch.FloatTensor([state_val for state_val in states_value]).to(self.device).reshape(states.shape[0],states.shape[1],-1)

	# ***********************************************************************************
		#update critic (value_net)
		curr_Q = self.value_network.forward(None,states_value)

		# discounted_rewards = np.asarray([[torch.sum(torch.FloatTensor([self.gamma**i for i in range(rewards[k][j:].size(0))])* rewards[k][j:]) for j in range(rewards.size(0))] for k in range(self.num_agents)])
		# discounted_rewards = np.transpose(discounted_rewards)
		discounted_rewards = self.calculate_returns(rewards,self.gamma)

	# ***********************************************************************************

	# ***********************************************************************************
		value_targets = rewards.to(self.device) + torch.FloatTensor(discounted_rewards).to(self.device)
		value_targets = value_targets.unsqueeze(dim=-1)
		value_loss = F.smooth_l1_loss(curr_Q,value_targets,reduction='none')

		critic_loss = torch.Tensor([[0] for i in range(self.num_agents)])
		weight_values = torch.Tensor([[0] for i in range(self.num_agents)])



		for values,weight_value in zip(value_loss,weight_action_clone):
			critic_loss+=values
			weight_values+=weight_value

		sum_weight_values = torch.sum(weight_values)

		for i in range(self.num_agents):
			critic_loss[i]+=self.lambda_*(sum_weight_values-weight_values[i])

		

		value_loss = torch.mean(critic_loss) / value_loss.shape[0]

	# ***********************************************************************************

		#update actor (policy net)
	# ***********************************************************************************

		entropy = -torch.mean(torch.sum(probs * torch.log(torch.clamp(probs, 1e-10,1.0)), dim=2))

		advantage = value_targets - curr_Q
		probs = Categorical(probs)
		policy_loss = -probs.log_prob(actions).unsqueeze(dim=-1) * advantage.detach()
		policy_loss = policy_loss.mean() - self.entropy_pen*entropy
	# ***********************************************************************************
		
	# ***********************************************************************************
		self.value_optimizer.zero_grad()
		value_loss.backward(retain_graph=False)
		grad_norm_value = torch.nn.utils.clip_grad_norm_(self.value_network.parameters(),0.5)
		self.value_optimizer.step()
		

		self.policy_optimizer.zero_grad()
		policy_loss.backward(retain_graph=False)
		grad_norm_policy = torch.nn.utils.clip_grad_norm_(self.policy_network.parameters(),0.5)
		self.policy_optimizer.step()
	# ***********************************************************************************
		return value_loss,policy_loss,entropy,grad_norm_value,grad_norm_policy




# 		'''
# 		Separate networks
# 		'''
		
# 	# ***********************************************************************************
# 		#update critic (value_net)
# 		curr_Q = self.value_network.forward(states)

		# # discounted_rewards = np.asarray([[torch.sum(torch.FloatTensor([self.gamma**i for i in range(rewards[k][j:].size(0))])* rewards[k][j:]) for j in range(rewards.size(0))] for k in range(self.num_agents)])
		# # discounted_rewards = np.transpose(discounted_rewards)
		# discounted_rewards = self.calculate_returns(rewards,self.gamma)
# 	# ***********************************************************************************

# 	# ***********************************************************************************
# 		value_targets = rewards.to(self.device) + torch.FloatTensor(discounted_rewards).to(self.device)
# 		value_targets = value_targets.unsqueeze(dim=-1)
# 		value_loss = F.smooth_l1_loss(curr_Q,value_targets)
# 	# ***********************************************************************************

# 		#update actor (policy net)
# 	# ***********************************************************************************
# 		probs = self.policy_network.forward(states)

# 		entropy = -torch.mean(torch.sum(probs * torch.log(torch.clamp(probs, 1e-10,1.0)), dim=2))

# 		advantage = value_targets - curr_Q
# 		probs = Categorical(probs)
# 		policy_loss = -probs.log_prob(actions).unsqueeze(dim=-1) * advantage.detach()
# 		policy_loss = policy_loss.mean() - self.entropy_pen*entropy
# 	# ***********************************************************************************
		
# 	# ***********************************************************************************
# 		self.value_optimizer.zero_grad()
# 		value_loss.backward(retain_graph=False)
# 		grad_norm_value = torch.nn.utils.clip_grad_norm_(self.value_network.parameters(),0.5)
# 		self.value_optimizer.step()
		

# 		self.policy_optimizer.zero_grad()
# 		policy_loss.backward(retain_graph=False)
# 		grad_norm_policy = torch.nn.utils.clip_grad_norm_(self.policy_network.parameters(),0.5)
# 		self.policy_optimizer.step()
# 	# ***********************************************************************************

# 		return value_loss,policy_loss,entropy,grad_norm_value,grad_norm_policy



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

