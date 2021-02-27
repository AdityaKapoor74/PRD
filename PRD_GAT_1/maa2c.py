import os
import torch
import torch.nn.functional as F 
import torch.optim as optim
from torch.distributions import Categorical
import torch.autograd as autograd
import numpy as np
from torch.utils.tensorboard import SummaryWriter
# from a2c_agent import A2CAgent
from a2c_agent_soft_attention import A2CAgent
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
			critic_dir = '../../models/Experiment8_3/critic_networks/'
			try: 
				os.makedirs(critic_dir, exist_ok = True) 
				print("Critic Directory created successfully") 
			except OSError as error: 
				print("Critic Directory can not be created") 
			actor_dir = '../../models/Experiment8_3/actor_networks/'
			try: 
				os.makedirs(actor_dir, exist_ok = True) 
				print("Actor Directory created successfully") 
			except OSError as error: 
				print("Actor Directory can not be created")  
			weight_dir = '../../weights/Experiment8_3/'
			try: 
				os.makedirs(weight_dir, exist_ok = True) 
				print("Weights Directory created successfully") 
			except OSError as error: 
				print("Weights Directory can not be created") 

			tensorboard_dir = '../../runs/Experiment8_3/'


			# paths for models, tensorboard and gifs
			self.critic_model_path = critic_dir+str(self.date_time)+'_VN_GAT1_PREPROC_GAT1_FC1_lr'+str(self.agents.value_lr)+'_PN_FC2_lr'+str(self.agents.policy_lr)+'_GradNorm0.5_Entropy'+str(self.agents.entropy_pen)+'_trace_decay'+str(self.agents.trace_decay)+'_lambda'+str(self.agents.lambda_)+'no_relu'
			self.actor_model_path = actor_dir+str(self.date_time)+'_PN_FC2_lr'+str(self.agents.policy_lr)+'_VN_GAT1_PREPROC_GAT1_FC1_lr'+str(self.agents.value_lr)+'_GradNorm0.5_Entropy'+str(self.agents.entropy_pen)+'_trace_decay'+str(self.agents.trace_decay)+'_lambda'+str(self.agents.lambda_)+'no_relu'
			self.tensorboard_path = tensorboard_dir+str(self.date_time)+'_VN_GAT1_PREPROC_GAT1_FC1_lr'+str(self.agents.value_lr)+'_PN_FC2_lr'+str(self.agents.policy_lr)+'_GradNorm0.5_Entropy'+str(self.agents.entropy_pen)+'_trace_decay'+str(self.agents.trace_decay)+'_lambda'+str(self.agents.lambda_)+'no_relu'
			self.filename = weight_dir+str(self.date_time)+'_VN_GAT1_PREPROC_GAT1_FC1_lr'+str(self.agents.value_lr)+'_PN_FC2_lr'+str(self.agents.policy_lr)+'_GradNorm0.5_Entropy'+str(self.agents.entropy_pen)+'_trace_decay'+str(self.agents.trace_decay)+'_lambda'+str(self.agents.lambda_)+'no_relu.txt'

		elif self.gif:
			gif_dir = '../../gifs/Experiment8_3/'
			try: 
				os.makedirs(gif_dir, exist_ok = True) 
				print("Gif Directory created successfully") 
			except OSError as error: 
				print("Gif Directory can not be created")
			self.gif_path = gif_dir+str(self.date_time)+'_VN_GAT1_PREPROC_GAT1_FC1_lr'+str(self.agents.value_lr)+'_PN_FC2_lr'+str(self.agents.policy_lr)+'_GradNorm0.5_Entropy'+str(self.agents.entropy_pen)+'_lambda'+str(self.agents.lambda_)+'.gif'

		if not(self.gif) and self.save:
			self.writer = SummaryWriter(self.tensorboard_path)
			
		self.src_edges_critic = []
		self.dest_edges_critic = []
		self.src_edges_actor = []
		self.dest_edges_actor = []

		for i in range(self.num_agents):
			for j in range(self.num_agents):
				self.src_edges_critic.append(i)
				self.dest_edges_critic.append(j)
				if i==j:
					continue
				self.src_edges_actor.append(i)
				self.dest_edges_actor.append(j)

		self.src_edges_critic = torch.tensor(self.src_edges_critic)
		self.dest_edges_critic = torch.tensor(self.dest_edges_critic)
		self.src_edges_actor = torch.tensor(self.src_edges_actor)
		self.dest_edges_actor = torch.tensor(self.dest_edges_actor)



	def get_actions(self,states):
		# actions = self.agents.get_action(actor_graph)
		# return actions
		actions = []
		for i in range(self.num_agents):
			action = self.agents.get_action(states[i])
			actions.append(action)
		return actions


	def calculate_weights(self,weights):
		paired_agents_weight = 0
		paired_agents_weight_count = 0
		unpaired_agents_weight = 0
		unpaired_agents_weight_count = 0

		for k in range(weights.shape[0]):
			for i in range(self.num_agents):
				for j in range(self.num_agents):
					if self.num_agents-1-i == j:
						paired_agents_weight += weights[k][i][j]
						paired_agents_weight_count += 1
					else:
						unpaired_agents_weight += weights[k][i][j]
						unpaired_agents_weight_count += 1

		return round(paired_agents_weight.item()/paired_agents_weight_count,4), round(unpaired_agents_weight.item()/unpaired_agents_weight_count,4)

	def calculate_metrics(self,weights,threshold):

		TP = [0]*self.num_agents
		FP = [0]*self.num_agents
		TN = [0]*self.num_agents
		FN = [0]*self.num_agents
		accuracy = [0]*self.num_agents
		precision = [0]*self.num_agents
		recall = [0]*self.num_agents

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

		# calculate accuracy/precision/recall before calculating rates
		for i in range(self.num_agents):
			TP_rate[i] = TP[i]/(TP[i]+FN[i])
			FP_rate[i] = FP[i]/(FP[i]+TN[i])
			TN_rate[i] = TN[i]/(TN[i]+FP[i])
			FN_rate[i] = FN[i]/(FN[i]+TP[i])

			if (TP[i]+TN[i]+FP[i]+FN[i]) == 0:
				accuracy[i] = 0
			else:
				accuracy[i] = round((TP[i]+TN[i])/(TP[i]+TN[i]+FP[i]+FN[i]),4)
			if (TP[i]+FP[i]) == 0:
				precision[i] = 0
			else:
				precision[i] = round((TP[i]/(TP[i]+FP[i])),4)
			if (TP[i]+FN[i]) == 0:
				recall[i] = 0
			else:
				recall[i] = round((TP[i]/(TP[i]+FN[i])),4)

		return TP_rate, FP_rate, TN_rate, FN_rate, accuracy, precision, recall


	def update(self,trajectory,episode):


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



		if not(self.gif) and self.save:
			for theta in [1e-5,1e-4,1e-3,1e-2,1e-1]:
				TP, FP, TN, FN, accuracy, precision, recall = self.calculate_metrics(weights,theta)

				for i in range(self.num_agents):
					self.writer.add_scalar('Weight Metric/TP (agent'+str(i)+') threshold:'+str(theta),TP[i],episode)
					self.writer.add_scalar('Weight Metric/FP (agent'+str(i)+') threshold:'+str(theta),FP[i],episode)
					self.writer.add_scalar('Weight Metric/TN (agent'+str(i)+') threshold:'+str(theta),TN[i],episode)
					self.writer.add_scalar('Weight Metric/FN (agent'+str(i)+') threshold:'+str(theta),FN[i],episode)
					self.writer.add_scalar('Weight Metric/Accuracy (agent'+str(i)+') threshold:'+str(theta),accuracy[i],episode)
					self.writer.add_scalar('Weight Metric/Precision (agent'+str(i)+') threshold:'+str(theta),precision[i],episode)
					self.writer.add_scalar('Weight Metric/Recall (agent'+str(i)+') threshold:'+str(theta),recall[i],episode)

				self.writer.add_scalar('Weight Metric/TP threshold:'+str(theta),sum(TP),episode)
				self.writer.add_scalar('Weight Metric/FP threshold:'+str(theta),sum(FP),episode)
				self.writer.add_scalar('Weight Metric/TN threshold:'+str(theta),sum(TN),episode)
				self.writer.add_scalar('Weight Metric/FN threshold:'+str(theta),sum(FN),episode)
				self.writer.add_scalar('Weight Metric/Accuracy threshold:'+str(theta),sum(accuracy),episode)
				self.writer.add_scalar('Weight Metric/Precision threshold:'+str(theta),sum(precision),episode)
				self.writer.add_scalar('Weight Metric/Recall threshold:'+str(theta),sum(recall),episode)




		if not(self.gif) and self.save:
			self.writer.add_scalar('Loss/Entropy loss',entropy.item(),episode)
			self.writer.add_scalar('Loss/Value Loss',value_loss.item(),episode)
			self.writer.add_scalar('Loss/Policy Loss',policy_loss.item(),episode)
			self.writer.add_scalar('Gradient Normalization/Grad Norm Value',grad_norm_value,episode)
			self.writer.add_scalar('Gradient Normalization/Grad Norm Policy',grad_norm_policy,episode)
			self.writer.add_scalar('Weights/Average Weights',torch.mean(weights).item(),episode)
			paired_agent_avg_weight, unpaired_agent_avg_weight = self.calculate_weights(weights)
			self.writer.add_scalar('Weights/Average Paired Agent Weights',paired_agent_avg_weight,episode)
			self.writer.add_scalar('Weights/Average Unpaired Agent Weights',unpaired_agent_avg_weight,episode)


			with open(self.filename,'a+') as f:
				torch.set_printoptions(profile="full")
				print("*"*50,file=f)
				print("EPISODE:",episode,file=f)
				print("*"*50,file=f)
				print(weights, file=f)
				torch.set_printoptions(profile="default")



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

		# graph = nx.complete_graph(self.num_agents)
		# graph = dgl.from_networkx(graph).to(self.device)
		# graph = dgl.transform.add_self_loop(graph)
		graph = dgl.graph((self.src_edges_critic,self.dest_edges_critic),idtype=torch.int32, device=self.device)

		graph.ndata['obs'] = torch.FloatTensor(states_critic).to(self.device)
		pose_goal = []
		for pose, goal in zip(states_critic[:,:2],states_critic[:,-2:]):
			pose_goal.append(np.concatenate([pose,goal]))
		graph.ndata['mypose_goalpose'] = torch.FloatTensor(pose_goal).to(self.device)
			   
		return graph


	def construct_agent_graph_actor(self,states_actor):

		# graph = nx.complete_graph(self.num_agents)
		# graph = dgl.from_networkx(graph).to(self.device)
		graph = dgl.graph((self.src_edges_actor,self.dest_edges_actor),idtype=torch.int32, device=self.device)

		graph.ndata['obs'] = torch.FloatTensor(states_actor).to(self.device)
			   
		return graph


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

			gif_checkpoint = 1

			trajectory = []
			episode_reward = 0
			for step in range(max_steps):

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
			if not(episode%1000) and episode!=0 and not(self.gif) and self.save:
				torch.save(self.agents.critic_network.state_dict(), self.critic_model_path+'_epsiode'+str(episode)+'.pt')
				torch.save(self.agents.policy_network.state_dict(), self.actor_model_path+'_epsiode'+str(episode)+'.pt')  

			if not(self.gif):
				self.update(trajectory,episode) 
			elif self.gif and not(episode%gif_checkpoint):
				print("GENERATING GIF")
				# self.make_gif(np.array(images),gif_file_name+str(episode),duration=len(images)*3*TIME_PER_STEP,true_image=True,salience=False)
				self.make_gif(np.array(images),self.gif_path)


