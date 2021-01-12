import torch
import torch.nn.functional as F 
import torch.optim as optim
from torch.distributions import Categorical
import torch.autograd as autograd
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from a2c_agent import A2CAgent
import datetime

import dgl
import networkx as nx

TIME_PER_STEP = 0.1

class MAA2C:

	def __init__(self,env, gif=True, save=True):
		# self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		self.device = "cpu"
		self.env = env
		self.gif = gif
		self.save = save
		self.num_agents = env.n
		self.num_actions = self.env.action_space[0].n
		self.one_hot_actions = torch.zeros(self.env.action_space[0].n,self.env.action_space[0].n)
		self.date_time = f"{datetime.datetime.now():%d-%m-%Y}"
		for i in range(self.env.action_space[0].n):
			self.one_hot_actions[i][i] = 1

		self.agents = A2CAgent(self.env, gif = self.gif)

		if not(self.gif) and self.save:
			self.writer = SummaryWriter('../../runs/GNN_V_values_i_j_weight_net_v2/4_agents/'+str(self.date_time)+'_VN_GNN2_GAT1_FC1_lr2e-4_PN_FC2_lr2e-4_GradNorm0.5_Entropy0.008_lambda1e-2_remote_mamba')

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

		TP_rate = [0]*self.num_agents
		FP_rate = [0]*self.num_agents
		TN_rate = [0]*self.num_agents
		FN_rate = [0]*self.num_agents

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
			TP_rate[i] = TP[i]/(TP[i]+FN[i])
			FP_rate[i] = FP[i]/(FP[i]+TN[i])
			TN_rate[i] = TN[i]/(TN[i]+FP[i])
			FN_rate[i] = FN[i]/(FN[i]+TP[i])
		return TP_rate, FP_rate, TN_rate, FN_rate


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

		for theta in [1e-5,1e-4,1e-3,1e-2,1e-1]:
			TP, FP, TN, FN = self.calculate_metrics(weights,theta,num_steps)

			if not(self.gif) and self.save:
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
						precision = round((TP[i]/(TP[i]+FP[i])),4)
					if (TP[i]+FP[i]) == 0:
						recall = 0
					else:
						recall = round((TP[i]/(TP[i]+FN[i])),4)
					self.writer.add_scalar('Weight Metric/TP (agent'+str(i)+') threshold:'+str(theta),TP[i],episode)
					self.writer.add_scalar('Weight Metric/FP (agent'+str(i)+') threshold:'+str(theta),FP[i],episode)
					self.writer.add_scalar('Weight Metric/TN (agent'+str(i)+') threshold:'+str(theta),TN[i],episode)
					self.writer.add_scalar('Weight Metric/FN (agent'+str(i)+') threshold:'+str(theta),FN[i],episode)
					self.writer.add_scalar('Weight Metric/Accuracy (agent'+str(i)+') threshold:'+str(theta),accuracy,episode)
					self.writer.add_scalar('Weight Metric/Precision (agent'+str(i)+') threshold:'+str(theta),precision,episode)
					self.writer.add_scalar('Weight Metric/Recall (agent'+str(i)+') threshold:'+str(theta),recall,episode)




		if not(self.gif) and self.save:
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


	# def make_gif(self, images, fname, duration=2, true_image=False,salience=False,salIMGS=None):
	# 	import moviepy.editor as mpy

	# 	def make_frame(t):
	# 		try:
	# 			x = images[int(len(images)/duration*t)]
	# 		except:
	# 			x = images[-1]

	# 		if true_image:
	# 			return x.astype(np.uint8)
	# 		else:
	# 			return ((x+1)/2*255).astype(np.uint8)

	# 	def make_mask(t):
	# 		try:
	# 			x = salIMGS[int(len(salIMGS)/duration*t)]
	# 		except:
	# 			x = salIMGS[-1]
	# 		return x

	# 	clip = mpy.VideoClip(make_frame, duration=duration)
	# 	if salience == True:
	# 		mask = mpy.VideoClip(make_mask, ismask=True,duration= duration)
	# 		clipB = clip.set_mask(mask)
	# 		clipB = clip.set_opacity(0)
	# 		mask = mask.set_opacity(0.1)
	# 		mask.write_gif(fname, fps = len(images) / duration,verbose=False)
	# 	else:
	# 		clip.write_gif(fname, fps = len(images) / duration,verbose=False)

	def make_gif(self,images,fname,fps=10, scale=1.0):
		from moviepy.editor import ImageSequenceClip
		"""Creates a gif given a stack of images using moviepy
		Notes
		-----
		works with current Github version of moviepy (not the pip version)
		https://github.com/Zulko/moviepy/commit/d4c9c37bc88261d8ed8b5d9b7c317d13b2cdf62e
		Usage
		-----
		>>> X = randn(100, 64, 64)
		>>> gif('test.gif', X)
		Parameters
		----------
		filename : string
			The filename of the gif to write to
		array : array_like
			A numpy array that contains a sequence of images
		fps : int
			frames per second (default: 10)
		scale : float
			how much to rescale each image by (default: 1.0)
		"""

		# copy into the color dimension if the images are black and white
		if images.ndim == 3:
			images = images[..., np.newaxis] * np.ones(3)

		# make the moviepy clip
		clip = ImageSequenceClip(list(images), fps=fps).resize(scale)
		clip.write_gif(fname, fps=fps)




	def run(self,max_episode,max_steps):  
		for episode in range(1,max_episode):
			states = self.env.reset()

			images = []

			states_critic,states_actor = self.split_states(states)

			gif_file_name = '../../gifs/GNN_V_values_i_j_weight_net_v2/4_agents/'+str(self.date_time)+'_VN_GNN2_GAT1_FC1_lr2e-4_PN_FC2_lr2e-4_GradNorm0.5_Entropy0.008_lambda1e-2_remote_mamba.gif'

			gif_checkpoint = 1

			trajectory = []
			episode_reward = 0
			end_step = 0
			for step in range(max_steps):

				# states_actor_graph = self.construct_agent_graph_actor(states_actor)

				# policies = self.agents.policy_network(torch.FloatTensor(states_actor).to(self.device)).detach().cpu().numpy()

				# actions = self.get_actions(states_actor_graph)

				if self.gif:
					# At each step, append an image to list
					images.append(np.squeeze(self.env.render(mode='rgb_array')))
					# Advance a step and render a new image
					with torch.no_grad():
						actions = self.get_actions(states_actor)
				else:
					actions = self.get_actions(states_actor)


				one_hot_actions = np.zeros((self.num_agents,self.num_actions))
				for i,act in enumerate(actions):
					one_hot_actions[i][act] = 1


				states_critic_graph = self.construct_agent_graph_critic(states_critic)



				next_states,rewards,dones,info = self.env.step(actions)
				next_states_critic,next_states_actor = self.split_states(next_states)

				episode_reward += np.sum(rewards)

				if not(self.gif):
					if all(dones) or step == max_steps-1:

						end_step = step

						dones = [1 for _ in range(self.num_agents)]
						trajectory.append([states_critic_graph,one_hot_actions,actions,states_actor,rewards,dones])
						print("*"*100)
						print("EPISODE: {} | REWARD: {} \n".format(episode,np.round(episode_reward,decimals=4)))
						print("*"*100)

						if not(self.gif) and self.save:
							self.writer.add_scalar('Reward Incurred/Length of the episode',step,episode)
							self.writer.add_scalar('Reward Incurred/Reward',episode_reward,episode)

						break
					else:
						dones = [0 for _ in range(self.num_agents)]
						trajectory.append([states_critic_graph,one_hot_actions,actions,states_actor,rewards,dones])
						states_critic,states_actor = next_states_critic,next_states_actor
						states = next_states

				else:
					states_critic,states_actor = next_states_critic,next_states_actor
					states = next_states


			#make a directory called models
			if not(episode%100) and episode!=0 and not(self.gif) and self.save:
				torch.save(self.agents.critic_network.state_dict(), "../../models/GNN_V_values_i_j_weight_net_v2/4_agents/critic_networks/"+str(self.date_time)+"_VN_GNN2_GAT1_FC1_lr2e-4_PN_FC2_lr2e-4_GradNorm0.5_Entropy0.008_lambda1e-2_remote_mamba"+str(episode)+".pt")
				torch.save(self.agents.policy_network.state_dict(), "../../models/GNN_V_values_i_j_weight_net_v2/4_agents/actor_networks/"+str(self.date_time)+"_PN_FC2_lr2e-4_VN_GNN2_GAT1_FC1_lr2e-4_GradNorm0.5_Entropy0.008_lambda1e-2_remote_mamba"+str(episode)+".pt")  

			if not(self.gif):
				self.update(trajectory,episode,end_step) 
			elif self.gif and not(episode%gif_checkpoint):
				print("GENERATING GIF")
				# self.make_gif(np.array(images),gif_file_name+str(episode),duration=len(images)*3*TIME_PER_STEP,true_image=True,salience=False)
				self.make_gif(np.array(images),gif_file_name)


