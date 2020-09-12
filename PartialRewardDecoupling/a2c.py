import torch
import torch.nn as nn
import torch.nn.functional as F

#**********************************************
# Action Conditioning with separate networks
#**********************************************
'''
Value Network: Weight Network + Action-Value Network
(Weight Network)
Input: Observations
Output: W1 and W2

(Action-Value Network)
Input: Observations+(Policy*W1+Action*W2) --- [Shape of Policy and Action are 1 x action_dim (1x5)]
Output: State-Action Value

Policy Network
Input: Observations
Output: Probability over all actions
'''

class ValueNetwork_(nn.Module):

	def __init__(self,input_states,num_agents,num_actions,output_dim_weights,output_dim_value):
		super(ValueNetwork_,self).__init__()

		self.num_agents = num_agents
		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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


	def forward(self,states,one_hot_actions,probs):

		weights = F.relu(self.weights_fc1(states))
		weights = F.relu(self.weights_fc2(weights))
		weights = self.final_weights(weights)
		weights = weights.reshape(weights.shape[0],weights.shape[1],self.num_agents-1,2)
		weights = self.softmax(weights)
		weights = weights.reshape(-1,2)

		weight_prob = weights[:,0]
		weight_action = weights[:,1]

		weight_prob = weight_prob.reshape(-1,self.num_agents,self.num_agents-1)
		weight_action = weight_action.reshape(-1,self.num_agents,self.num_agents-1)


		states_value = []
		for k in range(states.shape[0]):
			for j in range(states.shape[1]): # states.shape[1]==self.num_agents

				actions_ = one_hot_actions[k].detach().clone()
				actions_ = torch.cat([actions_[:j],actions_[j+1:]])
				probs_ = probs[k].detach().clone()
				probs_ = torch.cat([probs_[:j],probs_[j+1:]])


				z = torch.Tensor().to(self.device)
				for i in range(self.num_agents-1):
					z_partial = weight_action[k][j][i].item()*actions_[i].to(self.device)+weight_prob[k][j][i].item()*probs_[i]
					z = torch.cat([z,z_partial])

				z = z.reshape(self.num_agents-1,-1)
				tmp = states[k][j].clone()
				for i in range(self.num_agents-1):
					if i == self.num_agents-2:
						tmp = torch.cat([tmp,z[i]])
					else: 
						tmp = torch.cat([tmp[:-6*(self.num_agents-2-i)],z[i],tmp[-6*(self.num_agents-2-i):]])
				states_value.append(tmp)

		states_value = torch.stack(states_value).reshape(states.shape[0],states.shape[1],-1).to(self.device)

		value = F.relu(self.value_fc1(states_value))
		value = F.relu(self.value_fc2(value))
		value = self.value(value)

		return weight_action,weight_prob,value


class PolicyNetwork_(nn.Module):

	def __init__(self,input_dim,output_dim):
		super(PolicyNetwork_,self).__init__()
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
		return dists


#**********************************************
# Two headed Actor Critic
#**********************************************
'''
Value Network
Input: Observations
Output: State-Action value

Policy Network
Input: Observations
Output: Probability over action space

(Can have same optimizer)
'''

class TwoHeadedActorCritic(nn.Module):

	def __init__(self,value_input_dim,policy_input_dim,value_output_dim,policy_output_dim):
		super(ActorCritic,self).__init__()
		self.value_fc1 = nn.Linear(value_input_dim,512)
		torch.nn.init.xavier_uniform_(self.value_fc1.weight)
		# torch.nn.init.kaiming_normal_(self.value_fc1.weight, mode='fan_in')
		self.policy_fc1 = nn.Linear(policy_input_dim,512)
		torch.nn.init.xavier_uniform_(self.policy_fc1.weight)
		# torch.nn.init.kaiming_normal_(self.policy_fc1.weight, mode='fan_in')

		self.value_fc2 = nn.Linear(512,256)
		torch.nn.init.xavier_uniform_(self.value_fc2.weight)
		# torch.nn.init.kaiming_normal_(self.value_fc2.weight, mode='fan_in')
		self.policy_fc2 = nn.Linear(512,256)
		torch.nn.init.xavier_uniform_(self.policy_fc2.weight)
		# torch.nn.init.kaiming_normal_(self.policy_fc2.weight, mode='fan_in')

		self.value = nn.Linear(256,value_output_dim)
		torch.nn.init.xavier_uniform_(self.value.weight)
		# torch.nn.init.kaiming_normal_(self.value.weight, mode='fan_in')

		self.policy = nn.Linear(256,policy_output_dim)
		torch.nn.init.xavier_uniform_(self.policy.weight)
		# torch.nn.init.kaiming_normal_(self.policy.weight, mode='fan_in')

	def forward(self,state_value=None,state_policy=None):

		if state_value is not None:
			value = F.relu(self.value_fc1(state_value))
			value = F.relu(self.value_fc2(value))
			value = self.value(value)

		else:
			value = None

		if state_policy is not None:
			policy = F.relu(self.policy_fc1(state_policy))
			policy = F.relu(self.policy_fc2(policy))
			policy = self.policy(policy)
			policy = F.softmax(policy,dim=-1)
		else:
			policy = None

		return value,


#**********************************************
# Separate Actor-Critic Network
#**********************************************
'''
Value Network
Input: Observations
Output: State-Action value

Policy Network
Input: Observations
Output: Probability over action space

(Can have separate optimizers)
'''

class ValueNetwork(nn.Module):

    def __init__(self,input_dim,output_dim):
        super(ValueNetwork,self).__init__()
        self.fc1 = nn.Linear(input_dim,512)
        torch.nn.init.xavier_uniform_(self.fc1.weight)
        self.fc2 = nn.Linear(512,256)
        torch.nn.init.xavier_uniform_(self.fc2.weight)
        self.value = nn.Linear(256,output_dim)
        torch.nn.init.xavier_uniform_(self.value.weight)

    def forward(self,state):
        value = F.relu(self.fc1(state))
        value = F.relu(self.fc2(value))
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
        logits = F.softmax(logits,dim=-1)


        return logits


#**********************************************
# Shared ActorCritic
#**********************************************
'''
Input: Observations
Output: State-Action value + Probability distribution over actions
'''
class CentralizedActorCritic(nn.Module): 

    def __init__(self, obs_dim, value_output, policy_output):
        super(CentralizedActorCritic, self).__init__()

        self.obs_dim = obs_dim
        self.policy_output = policy_output
        self.value_output = value_output

        self.shared_layer_1 = nn.Linear(self.obs_dim, 512)
        # torch.nn.init.kaiming_normal_(self.shared_layer.weight, mode='fan_in')
        torch.nn.init.xavier_uniform_(self.shared_layer_1.weight)
        # torch.nn.init.xavier_normal_(self.shared_layer_1.weight)

        self.shared_layer_2 = nn.Linear(512, 256)
        # torch.nn.init.kaiming_normal_(self.shared_layer.weight, mode='fan_in')
        torch.nn.init.xavier_uniform_(self.shared_layer_2.weight)
        # torch.nn.init.xavier_normal_(self.shared_layer_2.weight)

        self.value = nn.Linear(256, self.value_output)
        # torch.nn.init.kaiming_normal_(self.value.weight, mode='fan_in')
        torch.nn.init.xavier_uniform_(self.value.weight)
        # torch.nn.init.xavier_normal_(self.value.weight)

        self.policy = nn.Linear(256, self.policy_output)
        # torch.nn.init.kaiming_normal_(self.policy.weight, mode='fan_in')
        torch.nn.init.xavier_uniform_(self.policy.weight)
        # torch.nn.init.xavier_normal_(self.policy.weight)

    def forward(self, x):
        x_s = F.relu(self.shared_layer_1(x))
        x_s = F.relu(self.shared_layer_2(x_s))

        qval = self.value(x_s)
        
        policy = self.policy(x_s)
        policy = F.softmax(policy,dim=-1)

        return qval,policy
#**********************************************
