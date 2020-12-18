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
			self.writer = SummaryWriter('../../runs/GNN_V_values_i_j_weight_net_v2/4_agents/value_2_layers_lr_2e-4_policy_lr_2e-4_entropy_0.008_policy_gnn')

	def get_actions(self,actor_graph):
		actions = self.agents.get_action(actor_graph)
		return actions
		# actions = []
		# for i in range(self.num_agents):
		# 	action = self.agents.get_action(states[i])
		# 	actions.append(action)
		# return actions

	def update(self,trajectory,episode):


		# critic_graphs = torch.FloatTensor([sars[0] for sars in trajectory]).to(self.device)
		# critic_graphs = torch.Tensor([sars[0] for sars in trajectory]).to(self.device)
		critic_graphs = [sars[0] for sars in trajectory]
		# critic_graphs = [item for sublist in critic_graphs for item in sublist]
		critic_graphs = dgl.batch(critic_graphs).to(self.device)
		policies = torch.FloatTensor([sars[1] for sars in trajectory]).to(self.device)
		one_hot_actions = torch.FloatTensor([sars[2] for sars in trajectory]).to(self.device)
		actions = torch.FloatTensor([sars[3] for sars in trajectory]).to(self.device)
		# states_actor = torch.FloatTensor([sars[4] for sars in trajectory]).to(self.device)
		actor_graphs = [sars[4] for sars in trajectory]
		actor_graphs = dgl.batch(actor_graphs).to(self.device)
		rewards = torch.FloatTensor([sars[5] for sars in trajectory]).to(self.device)
		dones = torch.FloatTensor([sars[6] for sars in trajectory])

		# value_loss,policy_loss,entropy,grad_norm_value,grad_norm_policy = self.agents.update(critic_graphs,policies.reshape(-1,self.num_actions),actions.reshape(-1,self.num_actions),states_actor,rewards,dones)
		value_loss,policy_loss,entropy,grad_norm_value,grad_norm_policy = self.agents.update(critic_graphs,policies,one_hot_actions,actions,actor_graphs,rewards,dones)


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



	def construct_agent_graph_critic(self,states_critic):

		graph = nx.complete_graph(self.num_agents)
		graph = dgl.from_networkx(graph).to(self.device)
		graph = dgl.transform.add_self_loop(graph)


		graph.ndata['obs'] = torch.FloatTensor(states_critic).to(self.device)
		       
		return graph


	def construct_agent_graph_actor(self,states_actor):

		graph = nx.complete_graph(self.num_agents)
		graph = dgl.from_networkx(graph).to(self.device)

		graph.ndata['obs'] = torch.FloatTensor(states_actor).to(self.device)
		       
		return graph



	def run(self,max_episode,max_steps):  
		for episode in range(1,max_episode):
			states = self.env.reset()

			states_critic,states_actor = self.split_states(states)


			trajectory = []
			episode_reward = 0
			for step in range(max_steps):

				states_actor_graph = self.construct_agent_graph_actor(states_actor)
				policies = self.agents.policy_network(states_actor_graph).detach().cpu().numpy()

				# policies = self.agents.policy_network(torch.FloatTensor(states_actor).to(self.device)).detach().cpu().numpy()

				actions = self.get_actions(states_actor_graph)
				# actions = self.get_actions(states_actor)
				one_hot_actions = np.zeros((self.num_agents,self.num_actions))
				for i,act in enumerate(actions):
					one_hot_actions[i][act] = 1


				states_critic_graph = self.construct_agent_graph_critic(states_critic)



				next_states,rewards,dones,info = self.env.step(actions)
				next_states_critic,next_states_actor = self.split_states(next_states)

				episode_reward += np.sum(rewards)


				if all(dones) or step == max_steps-1:

					dones = [1 for _ in range(self.num_agents)]
					trajectory.append([states_critic_graph,policies,one_hot_actions,actions,states_actor_graph,rewards,dones])
					print("*"*100)
					print("EPISODE: {} | REWARD: {} \n".format(episode,np.round(episode_reward,decimals=4)))
					print("*"*100)

					if not(self.gif):
						self.writer.add_scalar('Reward Incurred/Length of the episode',step,episode)
						self.writer.add_scalar('Reward Incurred/Reward',episode_reward,episode)

					break
				else:
					dones = [0 for _ in range(self.num_agents)]
					trajectory.append([states_critic_graph,policies,one_hot_actions,actions,states_actor_graph,rewards,dones])
					states_critic,states_actor = next_states_critic,next_states_actor
					states = next_states

			#make a directory called models
			if not(episode%100) and episode!=0 and not(self.gif):
				torch.save(self.agents.critic_network.state_dict(), "../../models/GNN_V_values_i_j_weight_net_v2/4_agents/value_net_2_layered_lr_2e-4_policy_lr_2e-4_with_grad_norm_0.5_entropy_pen_0.008_xavier_uniform_init_policy_gnn.pt")
				torch.save(self.agents.policy_network.state_dict(), "../../models/GNN_V_values_i_j_weight_net_v2/4_agents/policy_net_lr_2e-4_value_2_layered_lr_2e-4_with_grad_norm_0.5_entropy_pen_0.008_xavier_uniform_init_policy_gnn.pt")  


			self.update(trajectory,episode) 

