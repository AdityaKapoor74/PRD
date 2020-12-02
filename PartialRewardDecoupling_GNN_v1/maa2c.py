import torch
import torch.nn.functional as F 
import torch.optim as optim
from torch.distributions import Categorical
import torch.autograd as autograd
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from a2c_agent import A2CAgent


import dgl
import networkx as nx


class MAA2C:

	def __init__(self,env,gif=True):
		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		self.env = env
		self.gif = gif
		self.num_agents = env.n
		self.num_actions = self.env.action_space[0].n
		self.one_hot_actions = torch.zeros(self.env.action_space[0].n,self.env.action_space[0].n)
		for i in range(self.env.action_space[0].n):
			self.one_hot_actions[i][i] = 1

		self.agents = A2CAgent(self.env, gif = self.gif)

		if not(self.gif):
			self.writer = SummaryWriter('../../runs/V_values_i_j/4_agents/value_2_layers_lr_2e-4_policy_lr_2e-4_entropy_0.008_policies_and_actions')

	def get_actions(self,states):
		actions = []
		for i in range(self.num_agents):
			action = self.agents.get_action(states[i])
			actions.append(action)
		return actions

	def update(self,trajectory,episode):


		# critic_graphs = torch.FloatTensor([sars[0] for sars in trajectory]).to(self.device)
		# critic_graphs = torch.Tensor([sars[0] for sars in trajectory]).to(self.device)
		# critic_graphs = [sars[0] for sars in trajectory]
		critic_graphs = [sars[0] for sars in trajectory]
		actions = torch.FloatTensor([sars[2] for sars in trajectory]).to(self.device)
		rewards = torch.FloatTensor([sars[3] for sars in trajectory]).to(self.device)
		dones = torch.FloatTensor([sars[4] for sars in trajectory])

		value_loss,policy_loss,entropy,grad_norm_value,grad_norm_policy = self.agents.update(critic_graphs,states_actor,actions,rewards,dones)


		if not(self.gif):
			self.writer.add_scalar('Loss/Entropy loss',entropy,episode)
			self.writer.add_scalar('Loss/Value Loss',value_loss,episode)
			self.writer.add_scalar('Loss/Policy Loss',policy_loss,episode)
			self.writer.add_scalar('Gradient Normalization/Grad Norm Value',grad_norm_value,episode)
			self.writer.add_scalar('Gradient Normalization/Grad Norm Policy',grad_norm_policy,episode)



	def split_states(self,states):

		states_critic = []
		states_actor = []
		for i in range(self.num_agents):
			states_critic.append(states[i][0])
			states_actor.append(states[i][1])

		states_critic = np.asarray(states_critic)
		states_actor = np.asarray(states_actor)

		return states_critic,states_actor



	def construct_agent_graph(self,states_critic):

		graph = nx.complete_graph(self.num_agents)
		graph = dgl.from_networkx(graph).to(self.device)


		graph.ndata['obs'] = torch.FloatTensor(states_critic).to(self.device)
		       
		return graph





	def run(self,max_episode,max_steps):  
		for episode in range(1,max_episode):
			states = self.env.reset()

			states_critic,states_actor = self.split_states(states)


			trajectory = []
			episode_reward = 0
			for step in range(max_steps):
				# print("STATES CRITIC")
				# print(states_critic)
				'''
				agents observation with actions concatenated
				Number of Agents x Number of Agents x Observation Space concatenated with Actions taken or Policy adopted
				'''
				'''
				other agents with respect to every agent in question
				Number of Agents x Number of Agents x Number of Agents-1 x Observation Space concatenated with Actions taken or Policy adopted
				'''

				policies = self.agents.policy_network(torch.FloatTensor(states_actor).to(self.device)).detach().cpu().numpy()
				# print("POLICIES")
				# print(policies)

				actions = self.get_actions(states_actor)
				one_hot_actions = np.zeros((self.num_agents,self.num_actions))
				for i,act in enumerate(actions):
					one_hot_actions[i][act] = 1

				# print("ACTIONS")
				# print(one_hot_actions)

				states_action_policy_critic = np.zeros((self.num_agents,self.num_agents,states_critic.shape[1]+self.num_actions))

				for i in range(self.num_agents):
					for j in range(self.num_agents):
						if i==j:
							states_action_policy_critic[i][j] = np.concatenate([states_critic[j],one_hot_actions[j]])
						else:
							states_action_policy_critic[i][j] = np.concatenate([states_critic[j],policies[j]])

				# print("MIX")
				# print(states_action_policy_critic.shape)
				# print(states_action_policy_critic)

				# generate graphs for each agent pair
				store_graphs = []
				for i in range(self.num_agents):
					store_graphs.append(self.construct_agent_graph(states_action_policy_critic[i]))




				next_states,rewards,dones,info = self.env.step(actions)
				next_states_critic,next_states_actor = self.split_states(next_states)

				episode_reward += np.sum(rewards)


				if all(dones) or step == max_steps-1:

					dones = [1 for _ in range(self.num_agents)]
					trajectory.append([store_graphs,states_actor,actions,rewards,dones])
					print("*"*100)
					print("EPISODE: {} | REWARD: {} \n".format(episode,np.round(episode_reward,decimals=4)))
					print("*"*100)

					if not(self.gif):
						self.writer.add_scalar('Reward Incurred/Length of the episode',step,episode)
						self.writer.add_scalar('Reward Incurred/Reward',episode_reward,episode)

					break
				else:
					dones = [0 for _ in range(self.num_agents)]
					trajectory.append([store_graphs,states_actor,actions,rewards,dones])
					states_critic,states_actor = next_states_critic,next_states_actor
					states = next_states

			#make a directory called models
			if not(episode%100) and episode!=0 and not(self.gif):
				torch.save(self.agents.value_network.state_dict(), "../../models/V_values_i_j/4_agents/value_net_2_layered_lr_2e-4_policy_lr_2e-4_with_grad_norm_0.5_entropy_pen_0.008_xavier_uniform_init_policies_and_actions.pt")
				torch.save(self.agents.policy_network.state_dict(), "../../models/V_values_i_j/4_agents/policy_net_lr_2e-4_value_2_layered_lr_2e-4_with_grad_norm_0.5_entropy_pen_0.008_xavier_uniform_init_policies_and_actions.pt")  


			self.update(trajectory,episode) 

