import os
import torch
import torch.nn.functional as F 
import torch.optim as optim
from torch.distributions import Categorical
import torch.autograd as autograd
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from a2c_agent_collision_avoidance import A2CAgent
import datetime



class MAA2C:

	def __init__(self,env, gif=True, save=True):
		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		self.device = "cpu"
		self.env = env
		self.gif = gif
		self.save = save
		self.num_agents = env.n
		self.num_actions = self.env.action_space[0].n
		self.date_time = f"{datetime.datetime.now():%d-%m-%Y}"


		self.weight_dictionary = {}

		for i in range(self.num_agents):
			agent_name = 'agent %d' % i
			self.weight_dictionary[agent_name] = {}
			for j in range(self.num_agents):
				agent_name_ = 'agent %d' % j
				self.weight_dictionary[agent_name][agent_name_] = 0



		self.agents = A2CAgent(self.env, gif = self.gif)


		if not(self.gif) and self.save:
			critic_dir = '../../../models/Scalar_dot_product/collision_avoidance/8_Agents/SingleAttentionMechanism/without_prd/critic_networks/'
			try: 
				os.makedirs(critic_dir, exist_ok = True) 
				print("Critic Directory created successfully") 
			except OSError as error: 
				print("Critic Directory can not be created") 
			actor_dir = '../../../models/Scalar_dot_product/collision_avoidance/8_Agents/SingleAttentionMechanism/without_prd/actor_networks/'
			try: 
				os.makedirs(actor_dir, exist_ok = True) 
				print("Actor Directory created successfully") 
			except OSError as error: 
				print("Actor Directory can not be created")  
			weight_dir = '../../../weights/Scalar_dot_product/collision_avoidance/8_Agents/SingleAttentionMechanism/without_prd/'
			try: 
				os.makedirs(weight_dir, exist_ok = True) 
				print("Weights Directory created successfully") 
			except OSError as error: 
				print("Weights Directory can not be created") 

			tensorboard_dir = '../../../runs/Scalar_dot_product/collision_avoidance/8_Agents/SingleAttentionMechanism/without_prd/'


			# paths for models, tensorboard and gifs
			self.critic_model_path = critic_dir+str(self.date_time)+'VN_ATN_FCN_lr'+str(self.agents.value_lr)+'_PN_FCN_lr'+str(self.agents.policy_lr)+'_GradNorm0.5_Entropy'+str(self.agents.entropy_pen)+'_trace_decay'+str(self.agents.trace_decay)+"lambda_"+str(self.agents.lambda_)+"topK_"+str(self.agents.top_k)+"select_above_threshold"+str(self.agents.select_above_threshold)+"softmax_cut_threshold"+str(self.agents.softmax_cut_threshold)
			self.actor_model_path = actor_dir+str(self.date_time)+'_PN_FCN_lr'+str(self.agents.policy_lr)+'VN_SAT_FCN_lr'+str(self.agents.value_lr)+'_GradNorm0.5_Entropy'+str(self.agents.entropy_pen)+'_trace_decay'+str(self.agents.trace_decay)+"lambda_"+str(self.agents.lambda_)+"topK_"+str(self.agents.top_k)+"select_above_threshold"+str(self.agents.select_above_threshold)+"softmax_cut_threshold"+str(self.agents.softmax_cut_threshold)
			self.tensorboard_path = tensorboard_dir+str(self.date_time)+'VN_SAT_FCN_lr'+str(self.agents.value_lr)+'_PN_FCN_lr'+str(self.agents.policy_lr)+'_GradNorm0.5_Entropy'+str(self.agents.entropy_pen)+'_trace_decay'+str(self.agents.trace_decay)+"lambda_"+str(self.agents.lambda_)+"topK_"+str(self.agents.top_k)+"select_above_threshold"+str(self.agents.select_above_threshold)+"softmax_cut_threshold"+str(self.agents.softmax_cut_threshold)
			self.filename = weight_dir+str(self.date_time)+'VN_SAT_FCN_lr'+str(self.agents.value_lr)+'_PN_FCN_lr'+str(self.agents.policy_lr)+'_GradNorm0.5_Entropy'+str(self.agents.entropy_pen)+'_trace_decay'+str(self.agents.trace_decay)+"lambda_"+str(self.agents.lambda_)+"topK_"+str(self.agents.top_k)+"select_above_threshold"+str(self.agents.select_above_threshold)+"softmax_cut_threshold"+str(self.agents.softmax_cut_threshold)+'.txt'

		elif self.gif:
			gif_dir = '../../../gifs/Scalar_dot_product/collision_avoidance/8_Agents/SingleAttentionMechanism/without_prd/'
			try: 
				os.makedirs(gif_dir, exist_ok = True) 
				print("Gif Directory created successfully") 
			except OSError as error: 
				print("Gif Directory can not be created")
			self.gif_path = gif_dir+str(self.date_time)+'VN_SAT_FCN_lr'+str(self.agents.value_lr)+'_PN_FCN_lr'+str(self.agents.policy_lr)+'_GradNorm0.5_Entropy'+str(self.agents.entropy_pen)+"lambda_"+str(self.agents.lambda_)+"topK_"+str(self.agents.top_k)+"select_above_threshold"+str(self.agents.select_above_threshold)+"softmax_cut_threshold"+str(self.agents.softmax_cut_threshold)+'.gif'

		if not(self.gif) and self.save:
			self.writer = SummaryWriter(self.tensorboard_path)



	def get_actions(self,states):
		actions = []
		for i in range(self.num_agents):
			action = self.agents.get_action(states[i])
			actions.append(action)
		return actions


	def calculate_indiv_weights(self,weights):
		weights_per_agent = torch.sum(weights,dim=0) / weights.shape[0]

		for i in range(self.num_agents):
			agent_name = 'agent %d' % i
			for j in range(self.num_agents):
				agent_name_ = 'agent %d' % j
				self.weight_dictionary[agent_name][agent_name_] = weights_per_agent[i][j].item()



	def update(self,trajectory,episode):

		states_critic = torch.FloatTensor([sars[0] for sars in trajectory]).to(self.device)
		next_states_critic = torch.FloatTensor([sars[1] for sars in trajectory]).to(self.device)

		one_hot_actions = torch.FloatTensor([sars[2] for sars in trajectory]).to(self.device)
		one_hot_next_actions = torch.FloatTensor([sars[3] for sars in trajectory]).to(self.device)
		actions = torch.FloatTensor([sars[4] for sars in trajectory]).to(self.device)

		states_actor = torch.FloatTensor([sars[5] for sars in trajectory]).to(self.device)
		next_states_actor = torch.FloatTensor([sars[6] for sars in trajectory]).to(self.device)

		rewards = torch.FloatTensor([sars[7] for sars in trajectory]).to(self.device)
		dones = torch.FloatTensor([sars[8] for sars in trajectory]).to(self.device)
		
		value_loss,policy_loss,entropy,grad_norm_value,grad_norm_policy,weights = self.agents.update(states_critic,next_states_critic,one_hot_actions,one_hot_next_actions,actions,states_actor,next_states_actor,rewards,dones)



		if not(self.gif) and self.save:
			self.writer.add_scalar('Loss/Entropy loss',entropy.item(),episode)
			self.writer.add_scalar('Loss/Value Loss',value_loss.item(),episode)
			self.writer.add_scalar('Loss/Policy Loss',policy_loss.item(),episode)
			self.writer.add_scalar('Gradient Normalization/Grad Norm Value',grad_norm_value,episode)
			self.writer.add_scalar('Gradient Normalization/Grad Norm Policy',grad_norm_policy,episode)

			# self.calculate_indiv_weights(weights)
			# for i in range(self.num_agents):
			# 	agent_name = 'agent %d' % i
			# 	self.writer.add_scalars('Weights/Average_Weights/'+agent_name,self.weight_dictionary[agent_name],episode)

			# ENTROPY OF WEIGHTS
			entropy_weights = -torch.mean(torch.sum(weights * torch.log(torch.clamp(weights, 1e-10,1.0)), dim=2))
			self.writer.add_scalar('Weights/Entropy', entropy_weights.item(), episode)


	def split_states(self,states):

		states_critic = []
		states_actor = []
		for i in range(self.num_agents):
			states_critic.append(states[i][0])
			states_actor.append(states[i][1])

		states_critic = np.asarray(states_critic)
		states_actor = np.asarray(states_actor)

		return states_critic,states_actor



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

				next_states,rewards,dones,info = self.env.step(actions)
				next_states_critic,next_states_actor = self.split_states(next_states)

				# next actions
				next_actions = self.get_actions(next_states_actor)


				one_hot_next_actions = np.zeros((self.num_agents,self.num_actions))
				for i,act in enumerate(next_actions):
					one_hot_next_actions[i][act] = 1

				episode_reward += np.sum(rewards)

				if not(self.gif):
					if all(dones) or step == max_steps-1:

						trajectory.append([states_critic,next_states_critic,one_hot_actions,one_hot_next_actions,actions,states_actor,next_states_actor,rewards,dones])
						print("*"*100)
						print("EPISODE: {} | REWARD: {} | TIME TAKEN: {} / {} \n".format(episode,np.round(episode_reward,decimals=4),step+1,max_steps))
						print("*"*100)

						if not(self.gif) and self.save:
							self.writer.add_scalar('Reward Incurred/Length of the episode',step,episode)
							self.writer.add_scalar('Reward Incurred/Reward',episode_reward,episode)

						break
					else:
						trajectory.append([states_critic,next_states_critic,one_hot_actions,one_hot_next_actions,actions,states_actor,next_states_actor,rewards,dones])
						states_critic,states_actor = next_states_critic,next_states_actor
						states = next_states

				else:
					states_critic,states_actor = next_states_critic,next_states_actor
					states = next_states


			if not(episode%1000) and episode!=0 and not(self.gif) and self.save:
				torch.save(self.agents.critic_network.state_dict(), self.critic_model_path+'_epsiode'+str(episode)+'.pt')
				torch.save(self.agents.policy_network.state_dict(), self.actor_model_path+'_epsiode'+str(episode)+'.pt')  

			if not(self.gif):
				self.update(trajectory,episode) 
			elif self.gif and not(episode%gif_checkpoint):
				print("GENERATING GIF")
				self.make_gif(np.array(images),self.gif_path)