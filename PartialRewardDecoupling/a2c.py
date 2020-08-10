import torch
import torch.nn as nn
import torch.nn.functional as F
# from torch.distributions import Categorical

#**********************************************
# Two hidden layers
class ValueNetwork(nn.Module):

	def __init__(self,input_states,num_agents,num_actions,output_dim_weights,output_dim_value):
		super(ValueNetwork,self).__init__()

		self.softmax = nn.Softmax(dim=-1)

		self.weights_fc1 = nn.Linear(input_states,512)
		torch.nn.init.xavier_uniform_(self.weights_fc1.weight)
		self.weights_fc2 = nn.Linear(512,256)
		torch.nn.init.xavier_uniform_(self.weights_fc2.weight)
		self.final_weights = nn.Linear(256,output_dim_weights)
		torch.nn.init.xavier_uniform_(self.final_weights.weight)




		self.value_fc1 = nn.Linear(input_states+(num_agents-1)*num_actions,512)
		torch.nn.init.xavier_uniform_(self.value_fc1.weight)
		self.value_fc2 = nn.Linear(512,256)
		torch.nn.init.xavier_uniform_(self.value_fc2.weight)
		self.value = nn.Linear(256,output_dim_value)
		torch.nn.init.xavier_uniform_(self.value.weight)


	def forward(self,state_weight,state_value):

		if state_weight is not None:
			weights = F.relu(self.weights_fc1(state_weight))
			weights = F.relu(self.weights_fc2(weights))
			weights = self.softmax(self.final_weights(weights))
			return weights

		if state_value is not None:
			value = F.relu(self.value_fc1(state_value))
			value = F.relu(self.value_fc2(value))
			value = self.value(value)
			return value


class PolicyNetwork(nn.Module):

	def __init__(self,input_dim,output_dim):
		super(PolicyNetwork,self).__init__()
		self.fc1 = nn.Linear(input_dim,512)
		torch.nn.init.xavier_uniform_(self.fc1.weight)
		self.fc2 = nn.Linear(512,256)
		torch.nn.init.xavier_uniform_(self.fc2.weight)
		self.policy = nn.Linear(256,output_dim)
		torch.nn.init.xavier_uniform_(self.policy.weight)

	def forward(self,state):
		logits = F.relu(self.fc1(state))
		logits = F.relu(self.fc2(logits))
		logits = self.policy(logits)
		dists = F.softmax(logits,dim=-1)
		# probs = Categorical(dists)

		# return probs
		return dists


# class WeightNetwork(nn.Module):

# 	def __init__(self,input_dim,output_dim):
# 		super(WeightNetwork,self).__init__()
# 		self.fc1 = nn.Linear(input_dim,512)
# 		torch.nn.init.xavier_uniform_(self.fc1.weight)
# 		self.fc2 = nn.Linear(512,256)
# 		torch.nn.init.xavier_uniform_(self.fc2.weight)
# 		self.weights = nn.Linear(256,output_dim)
# 		torch.nn.init.xavier_uniform_(self.weights.weight)

# 	def forward(self,state):
# 		weight = F.relu(self.fc1(state))
# 		weight = F.relu(self.fc2(weight))
# 		weight = self.weights(weight)

# 		return weight
#**********************************************

#**********************************************
# Two headed Actor-Critic
# class ActorCritic(nn.Module):

# 	def __init__(self,value_input_dim,policy_input_dim,value_output_dim,policy_output_dim):
# 		super(ActorCritic,self).__init__()
# 		self.value_fc1 = nn.Linear(value_input_dim,512)
# 		# torch.nn.init.xavier_uniform_(self.value_fc1.weight)
# 		torch.nn.init.kaiming_normal_(self.value_fc1.weight, mode='fan_in')
# 		self.policy_fc1 = nn.Linear(policy_input_dim,512)
# 		# torch.nn.init.xavier_uniform_(self.policy_fc1.weight)
# 		torch.nn.init.kaiming_normal_(self.policy_fc1.weight, mode='fan_in')

# 		self.value_fc2 = nn.Linear(512,256)
# 		# torch.nn.init.xavier_uniform_(self.value_fc2.weight)
# 		torch.nn.init.kaiming_normal_(self.value_fc2.weight, mode='fan_in')
# 		self.policy_fc2 = nn.Linear(512,256)
# 		# torch.nn.init.xavier_uniform_(self.policy_fc2.weight)
# 		torch.nn.init.kaiming_normal_(self.policy_fc2.weight, mode='fan_in')

# 		self.value = nn.Linear(256,value_output_dim)
# 		# torch.nn.init.xavier_uniform_(self.value.weight)
# 		torch.nn.init.kaiming_normal_(self.value.weight, mode='fan_in')
# 		self.policy = nn.Linear(256,policy_output_dim)
# 		# torch.nn.init.xavier_uniform_(self.policy.weight)
# 		torch.nn.init.kaiming_normal_(self.policy.weight, mode='fan_in')

# 	def forward(self,state_value=None,state_policy=None):

# 		if state_value is not None:
# 			value = F.relu(self.value_fc1(state_value))
# 			value = F.relu(self.value_fc2(value))
# 			value = self.value(value)

# 		else:
# 			value = None

# 		if state_policy is not None:
# 			policy = F.relu(self.policy_fc1(state_policy))
# 			policy = F.relu(self.policy_fc2(policy))
# 			policy = self.policy(policy)
# 		else:
# 			policy = None

# 		return value,policy
#**********************************************


#**********************************************
# One hidden layer
# class ValueNetwork(nn.Module):

#     def __init__(self,input_dim,output_dim):
#         super(ValueNetwork,self).__init__()
#         self.fc1 = nn.Linear(input_dim,256)
#         torch.nn.init.xavier_uniform_(self.fc1.weight)
#         self.value = nn.Linear(256,output_dim)
#         torch.nn.init.xavier_uniform_(self.value.weight)

#     def forward(self,state):
#         value = F.relu(self.fc1(state))
#         value = self.value(value)

#         return value


# class PolicyNetwork(nn.Module):

#     def __init__(self,input_dim,output_dim):
#         super(PolicyNetwork,self).__init__()
#         self.fc1 = nn.Linear(input_dim,256)
#         torch.nn.init.xavier_uniform_(self.fc1.weight)
#         self.policy = nn.Linear(256,output_dim)
#         torch.nn.init.xavier_uniform_(self.policy.weight)

#     def forward(self,state):
#         logits = F.relu(self.fc1(state))
#         logits = self.policy(logits)

#         return logits

#**********************************************

# Shared ActorCritic
#**********************************************
# class CentralizedActorCritic(nn.Module): 

#     def __init__(self, obs_dim, action_dim):
#         super(CentralizedActorCritic, self).__init__()

#         self.obs_dim = obs_dim
#         self.action_dim = action_dim

#         self.shared_layer_1 = nn.Linear(self.obs_dim, 512)
#         # torch.nn.init.kaiming_normal_(self.shared_layer.weight, mode='fan_in')
#         torch.nn.init.xavier_uniform_(self.shared_layer_1.weight)
#         # torch.nn.init.xavier_normal_(self.shared_layer_1.weight)

#         self.shared_layer_2 = nn.Linear(512, 256)
#         # torch.nn.init.kaiming_normal_(self.shared_layer.weight, mode='fan_in')
#         torch.nn.init.xavier_uniform_(self.shared_layer_2.weight)
#         # torch.nn.init.xavier_normal_(self.shared_layer_2.weight)

#         self.value = nn.Linear(256, 1)
#         # torch.nn.init.kaiming_normal_(self.value.weight, mode='fan_in')
#         torch.nn.init.xavier_uniform_(self.value.weight)
#         # torch.nn.init.xavier_normal_(self.value.weight)

#         self.policy = nn.Linear(256, self.action_dim)
#         # torch.nn.init.kaiming_normal_(self.policy.weight, mode='fan_in')
#         torch.nn.init.xavier_uniform_(self.policy.weight)
#         # torch.nn.init.xavier_normal_(self.policy.weight)

#     def forward(self, x):
#         x_s = F.relu(self.shared_layer_1(x))
#         x_s = F.relu(self.shared_layer_2(x_s))
#         qval = self.value(x_s)
#         policy = self.policy(x_s)

#         return qval,policy
#**********************************************
