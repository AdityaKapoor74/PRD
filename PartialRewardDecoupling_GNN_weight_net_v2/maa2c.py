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
		# self.device = "cpu"
		self.env = env
		self.gif = gif
		self.num_agents = env.n
		self.num_actions = self.env.action_space[0].n
		self.one_hot_actions = torch.zeros(self.env.action_space[0].n,self.env.action_space[0].n)
		for i in range(self.env.action_space[0].n):
			self.one_hot_actions[i][i] = 1

		self.agents = A2CAgent(self.env, gif = self.gif)

		if not(self.gif):
			self.writer = SummaryWriter('../../runs/GNN_V_values_i_j_weight_net_v2/4_agents/28_12_2020_VN_GNN2_GAT1_FC1_lr2e-4_PN_FC2_lr2e-4_GradNorm0.5_Entropy0.008_lambda1e-4_remote_mamba')

	def get_actions(self,states):
		# actions = self.agents.get_action(actor_graph)
		# return actions
		actions = []
		for i in range(self.num_agents):
			action = self.agents.get_action(states[i])
			actions.append(action)
		return actions


	def calculate_metrics(self,weights,threshold,num_steps):

		TP = [0]*self.num_agents
		FP = [0]*self.num_agents
		TN = [0]*self.num_agents
		FN = [0]*self.num_agents

		for k in range(weights.shape[0]):
			for i in range(self.num_agents):
				for j in range(self.num_agents):
					if self.num_agents-1-i == j:
						if weights[k][i][j] >= threshold:
							TP[i] += 1
						else:
							FN[i] += 1
					else:
						if weights[k][i][j] >= threshold:
							FP[i] += 1
						else:
							TN[i] += 1
		for i in range(self.num_agents):
			TP[i] = TP[i]/num_steps
			FP[i] = FP[i]/num_steps
			TN[i] = TN[i]/num_steps
			FN[i] = FN[i]/num_steps
		return TP, FP, TN, FN


	def update(self,trajectory,episode,num_steps):


		# critic_graphs = torch.FloatTensor([sars[0] for sars in trajectory]).to(self.device)
		# critic_graphs = torch.Tensor([sars[0] for sars in trajectory]).to(self.device)
		critic_graphs = [sars[0] for sars in trajectory]
		# critic_graphs = [item for sublist in critic_graphs for item in sublist]
		critic_graphs = dgl.batch(critic_graphs).to(self.device)
		one_hot_actions = torch.FloatTensor([sars[1] for sars in trajectory]).to(self.device)
		actions = torch.FloatTensor([sars[2] for sars in trajectory]).to(self.device)
		states_actor = torch.FloatTensor([sars[3] for sars in trajectory]).to(self.device)
		# actor_graphs = [sars[3] for sars in trajectory]
		# actor_graphs = dgl.batch(actor_graphs).to(self.device)
		rewards = torch.FloatTensor([sars[4] for sars in trajectory]).to(self.device)
		dones = torch.FloatTensor([sars[5] for sars in trajectory])

		# value_loss,policy_loss,entropy,grad_norm_value,grad_norm_policy = self.agents.update(critic_graphs,policies.reshape(-1,self.num_actions),actions.reshape(-1,self.num_actions),states_actor,rewards,dones)
		value_loss,policy_loss,entropy,grad_norm_value,grad_norm_policy,weights = self.agents.update(critic_graphs,one_hot_actions,actions,states_actor,rewards,dones)

		for theta in [1e-5,1e-4,1e-3,1e-2,1e-1,1]:
			TP, FP, TN, FN = self.calculate_metrics(weights,theta,num_steps)

			if not(self.gif):
				for i in range(self.num_agents):
					accuracy = 0
					precision = 0
					recall = 0
					if (TP[i]+TN[i]+FP[i]+FN[i]) == 0:
						accuracy = 0
					else:
						accuracy = round((TP[i]+TN[i])/(TP[i]+TN[i]+FP[i]+FN[i]),4)
					if (TP[i]+FN[i]) == 0:
						precision = 0
					else:
						precision = round((TP[i]/(TP[i]+FN[i])),4)
					if (TP[i]+FP[i]) == 0:
						recall = 0
					else:
						recall = round((TP[i]/(TP[i]+FP[i])),4)
					self.writer.add_scalar('Weight Metric/TP (agent'+str(i)+') threshold:'+str(theta),TP[i],episode)
					self.writer.add_scalar('Weight Metric/FP (agent'+str(i)+') threshold:'+str(theta),FP[i],episode)
					self.writer.add_scalar('Weight Metric/TN (agent'+str(i)+') threshold:'+str(theta),TN[i],episode)
					self.writer.add_scalar('Weight Metric/FN (agent'+str(i)+') threshold:'+str(theta),FN[i],episode)
					self.writer.add_scalar('Weight Metric/Accuracy (agent'+str(i)+') threshold:'+str(theta),accuracy,episode)
					self.writer.add_scalar('Weight Metric/Precision (agent'+str(i)+') threshold:'+str(theta),precision,episode)
					self.writer.add_scalar('Weight Metric/Recall (agent'+str(i)+') threshold:'+str(theta),recall,episode)




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
			end_step = 0
			for step in range(max_steps):

				# states_actor_graph = self.construct_agent_graph_actor(states_actor)

				# policies = self.agents.policy_network(torch.FloatTensor(states_actor).to(self.device)).detach().cpu().numpy()

				# actions = self.get_actions(states_actor_graph)
				actions = self.get_actions(states_actor)
				one_hot_actions = np.zeros((self.num_agents,self.num_actions))
				for i,act in enumerate(actions):
					one_hot_actions[i][act] = 1


				states_critic_graph = self.construct_agent_graph_critic(states_critic)



				next_states,rewards,dones,info = self.env.step(actions)
				next_states_critic,next_states_actor = self.split_states(next_states)

				episode_reward += np.sum(rewards)


				if all(dones) or step == max_steps-1:

					end_step = step

					dones = [1 for _ in range(self.num_agents)]
					trajectory.append([states_critic_graph,one_hot_actions,actions,states_actor,rewards,dones])
					print("*"*100)
					print("EPISODE: {} | REWARD: {} \n".format(episode,np.round(episode_reward,decimals=4)))
					print("*"*100)

					if not(self.gif):
						self.writer.add_scalar('Reward Incurred/Length of the episode',step,episode)
						self.writer.add_scalar('Reward Incurred/Reward',episode_reward,episode)

					break
				else:
					dones = [0 for _ in range(self.num_agents)]
					trajectory.append([states_critic_graph,one_hot_actions,actions,states_actor,rewards,dones])
					states_critic,states_actor = next_states_critic,next_states_actor
					states = next_states

			#make a directory called models
			if not(episode%100) and episode!=0 and not(self.gif):
				torch.save(self.agents.critic_network.state_dict(), "../../models/GNN_V_values_i_j_weight_net_v2/4_agents/critic_networks/28_12_2020_VN_GNN2_GAT1_FC1_lr2e-4_PN_FC2_lr2e-4_GradNorm0.5_Entropy0.008_lambda1e-4_remote_mamba"+str(episode)+".pt")
				torch.save(self.agents.policy_network.state_dict(), "../../models/GNN_V_values_i_j_weight_net_v2/4_agents/actor_networks/28_12_2020PN_FC2_lr2e-4_VN_GNN2_GAT1_FC1_lr2e-4_GradNorm0.5_Entropy0.008_lambda1e-4_remote_mamba"+str(episode)+".pt")  


			self.update(trajectory,episode,end_step) 

