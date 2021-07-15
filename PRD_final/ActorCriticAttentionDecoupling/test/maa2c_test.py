import os
import torch
import torch.nn.functional as F 
import torch.optim as optim
from torch.distributions import Categorical
import torch.autograd as autograd
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from a2c_agent_test import A2CAgent
import datetime



class MAA2C:

	def __init__(self, env, dictionary):
		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		self.env = env
		self.gif = dictionary["gif"]
		self.save = dictionary["save"]
		self.critic_type = dictionary["critic_type"]
		self.learn = dictionary["learn"]
		self.gif_checkpoint = dictionary["gif_checkpoint"]
		self.num_agents = env.n
		self.num_actions = self.env.action_space[0].n
		self.attention_heads = dictionary["attention_heads"]
		self.date_time = f"{datetime.datetime.now():%d-%m-%Y}"
		self.env_name = dictionary["env"]

		self.max_episodes = dictionary["max_episodes"]
		self.max_time_steps = dictionary["max_time_steps"]

		self.agents = A2CAgent(self.env, dictionary)

		self.weight_dictionary = {}

		for i in range(self.num_agents):
			agent_name = 'agent %d' % i
			self.weight_dictionary[agent_name] = {}
			for j in range(self.num_agents):
				agent_name_ = 'agent %d' % j
				self.weight_dictionary[agent_name][agent_name_] = 0

		self.weight_ent_per_head = {}
		for i in range(self.attention_heads):
			head_name = 'head %d' % i
			self.weight_ent_per_head[head_name] = 0

		self.value_loss = {"MLP_CRITIC_STATE":None, "MLP_CRITIC_STATE_ACTION":None, "GNN_CRITIC_STATE":None, "GNN_CRITIC_STATE_ACTION":None}
		self.critic_weights_entropy = {"MLP_CRITIC_STATE":None, "MLP_CRITIC_STATE_ACTION":None, "GNN_CRITIC_STATE":None, "GNN_CRITIC_STATE_ACTION":None}
		self.grad_norm_value = {"MLP_CRITIC_STATE":None, "MLP_CRITIC_STATE_ACTION":None, "GNN_CRITIC_STATE":None, "GNN_CRITIC_STATE_ACTION":None}
		self.critic = ["MLP_CRITIC_STATE", "MLP_CRITIC_STATE_ACTION", "GNN_CRITIC_STATE", "GNN_CRITIC_STATE_ACTION"]

		# MLP TO GNN
		self.value_loss_ = {"MLPToGNNV1":None, "MLPToGNNV2":None, "MLPToGNNV3":None, "MLPToGNNV4":None, "MLPToGNNV5":None, "MLPToGNNV6":None, "MLPToGNNV7":None, "MLPToGNNV8":None}
		self.critic_weights_entropy_ = {"MLPToGNNV1":None, "MLPToGNNV2":None, "MLPToGNNV3":None, "MLPToGNNV4":None, "MLPToGNNV5":None, "MLPToGNNV6":None, "MLPToGNNV7":None, "MLPToGNNV8_preproc":None, "MLPToGNNV8":None}
		self.grad_norm_value_ = {"MLPToGNNV1":None, "MLPToGNNV2":None, "MLPToGNNV3":None, "MLPToGNNV4":None, "MLPToGNNV5":None, "MLPToGNNV6":None, "MLPToGNNV7":None, "MLPToGNNV8":None}
		self.critic_ = ["MLPToGNNV1", "MLPToGNNV2", "MLPToGNNV3", "MLPToGNNV4", "MLPToGNNV5", "MLPToGNNV6", "MLPToGNNV7", "MLPToGNNV8_preproc", "MLPToGNNV8"]

		# DUAL
		self.value_loss_dual = {"Critic1":None, "Critic2":None}
		self.critic_weights_entropy_dual = {"Critic1":None, "Critic2":None}
		self.grad_norm_value_dual = {"Critic1":None, "Critic2":None}
		self.critic_dual = ["Critic1", "Critic2"]

		if self.save:
			critic_dir = dictionary["critic_dir"]
			try: 
				os.makedirs(critic_dir, exist_ok = True) 
				print("Critic Directory created successfully") 
			except OSError as error: 
				print("Critic Directory can not be created") 
			actor_dir = dictionary["actor_dir"]
			try: 
				os.makedirs(actor_dir, exist_ok = True) 
				print("Actor Directory created successfully") 
			except OSError as error: 
				print("Actor Directory can not be created")

			tensorboard_dir = dictionary["tensorboard_dir"]


			# paths for models, tensorboard and gifs
			self.critic_model_path = critic_dir+str(self.date_time)+'VN_ATN_FCN_lr'+str(self.agents.value_lr)+'_PN_ATN_FCN_lr'+str(self.agents.policy_lr)+'_GradNorm0.5_Entropy'+str(self.agents.entropy_pen)+'_trace_decay'+str(self.agents.trace_decay)+"topK_"+str(self.agents.top_k)+"select_above_threshold"+str(self.agents.select_above_threshold)+"softmax_cut_threshold"+str(self.agents.softmax_cut_threshold)
			self.actor_model_path = actor_dir+str(self.date_time)+'_PN_ATN_FCN_lr'+str(self.agents.policy_lr)+'VN_SAT_FCN_lr'+str(self.agents.value_lr)+'_GradNorm0.5_Entropy'+str(self.agents.entropy_pen)+'_trace_decay'+str(self.agents.trace_decay)+"topK_"+str(self.agents.top_k)+"select_above_threshold"+str(self.agents.select_above_threshold)+"softmax_cut_threshold"+str(self.agents.softmax_cut_threshold)
			self.tensorboard_path = tensorboard_dir+str(self.date_time)+'VN_SAT_FCN_lr'+str(self.agents.value_lr)+'_PN_ATN_FCN_lr'+str(self.agents.policy_lr)+'_GradNorm0.5_Entropy'+str(self.agents.entropy_pen)+'_trace_decay'+str(self.agents.trace_decay)+"topK_"+str(self.agents.top_k)+"select_above_threshold"+str(self.agents.select_above_threshold)+"softmax_cut_threshold"+str(self.agents.softmax_cut_threshold)
	
		elif self.gif:
			gif_dir = dictionary["gif_dir"]
			try: 
				os.makedirs(gif_dir, exist_ok = True) 
				print("Gif Directory created successfully") 
			except OSError as error: 
				print("Gif Directory can not be created")
			self.gif_path = gif_dir+str(self.date_time)+'VN_SAT_FCN_lr'+str(self.agents.value_lr)+'_PN_ATN_FCN_lr'+str(self.agents.policy_lr)+'_GradNorm0.5_Entropy'+str(self.agents.entropy_pen)+"topK_"+str(self.agents.top_k)+"select_above_threshold"+str(self.agents.select_above_threshold)+"softmax_cut_threshold"+str(self.agents.softmax_cut_threshold)+'.gif'

		if self.save:
			self.writer = SummaryWriter(self.tensorboard_path)

	def get_actions(self,states):
		actions = self.agents.get_action(states)
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
		
		value_loss,policy_loss,entropy,grad_norm_value,grad_norm_policy,weights,weight_policy = self.agents.update(states_critic,next_states_critic,one_hot_actions,one_hot_next_actions,actions,states_actor,next_states_actor,rewards,dones)

		if self.save:
			if self.critic_type == "ALL":
				for i,name in enumerate(self.critic):
					self.value_loss[name] = value_loss[i].item()
					self.grad_norm_value[name] = grad_norm_value[i]
				self.writer.add_scalars('Loss/Value Loss',self.value_loss,episode)
				self.writer.add_scalars('Gradient Normalization/Grad Norm Value',self.grad_norm_value,episode)

			elif self.critic_type == "ALL_W_POL":
				for i,name in enumerate(self.critic):
					self.value_loss[name] = value_loss[i].item()
					self.grad_norm_value[name] = grad_norm_value[i]
					# self.critic_weights_entropy[name] = -torch.mean(torch.sum(weights[i] * torch.log(torch.clamp(weights[i], 1e-10,1.0)), dim=2)).item()
				self.writer.add_scalars('Loss/Value Loss',self.value_loss,episode)
				self.writer.add_scalars('Gradient Normalization/Grad Norm Value',self.grad_norm_value,episode)
				# self.writer.add_scalars('Weights_Critic/Entropy', self.critic_weights_entropy, episode)
				self.writer.add_scalar('Loss/Entropy loss',entropy.item(),episode)
				self.writer.add_scalar('Loss/Policy Loss',policy_loss.item(),episode)
				self.writer.add_scalar('Gradient Normalization/Grad Norm Policy',grad_norm_policy,episode)
			elif self.critic_type == "MLPToGNN":
				for i,name in enumerate(self.critic_):
					self.critic_weights_entropy_[name] = -torch.mean(torch.sum(weights[i] * torch.log(torch.clamp(weights[i], 1e-10,1.0)), dim=2)).item()
					if name == "MLPToGNNV8_preproc":
						continue
					self.value_loss_[name] = value_loss[i].item()
					self.grad_norm_value_[name] = grad_norm_value[i]
				self.writer.add_scalars('Loss/Value Loss',self.value_loss_,episode)
				self.writer.add_scalars('Gradient Normalization/Grad Norm Value',self.grad_norm_value_,episode)
				self.writer.add_scalars('Weights_Critic/Entropy', self.critic_weights_entropy_, episode)
				self.writer.add_scalar('Loss/Entropy loss',entropy.item(),episode)
				self.writer.add_scalar('Loss/Policy Loss',policy_loss.item(),episode)
				self.writer.add_scalar('Gradient Normalization/Grad Norm Policy',grad_norm_policy,episode)
			elif "AttentionCritic" in self.critic_type:
				self.writer.add_scalar('Loss/Entropy loss',entropy.item(),episode)
				self.writer.add_scalar('Loss/Value Loss',value_loss.item(),episode)
				self.writer.add_scalar('Loss/Policy Loss',policy_loss.item(),episode)
				self.writer.add_scalar('Gradient Normalization/Grad Norm Value',grad_norm_value,episode)
				self.writer.add_scalar('Gradient Normalization/Grad Norm Policy',grad_norm_policy,episode)

				for i in range(self.attention_heads):
					head_name = 'head %d' % i
					self.weight_ent_per_head[head_name] = -torch.mean(torch.sum(weights[i] * torch.log(torch.clamp(weights[i], 1e-10,1.0)), dim=2)).item()
				self.writer.add_scalars('Weights_Critic/Entropy', self.weight_ent_per_head, episode)
			elif self.critic_type == "MultiHead":
				self.writer.add_scalar('Loss/Entropy loss',entropy.item(),episode)
				self.writer.add_scalar('Loss/Value Loss',value_loss.item(),episode)
				self.writer.add_scalar('Loss/Policy Loss',policy_loss.item(),episode)
				self.writer.add_scalar('Gradient Normalization/Grad Norm Value',grad_norm_value,episode)
				self.writer.add_scalar('Gradient Normalization/Grad Norm Policy',grad_norm_policy,episode)

				for i in range(self.attention_heads):
					head_name = 'head %d' % i
					self.weight_ent_per_head[head_name] = -torch.mean(torch.sum(weights[i] * torch.log(torch.clamp(weights[i], 1e-10,1.0)), dim=2)).item()
				self.writer.add_scalars('Weights_Critic/Entropy', self.weight_ent_per_head, episode)
			elif "Dual" in self.critic_type:
				for i,name in enumerate(self.critic_dual):
					self.value_loss_dual[name] = value_loss[i].item()
					self.grad_norm_value_dual[name] = grad_norm_value[i]
					self.critic_weights_entropy_dual[name] = -torch.mean(torch.sum(weights[i] * torch.log(torch.clamp(weights[i], 1e-10,1.0)), dim=2)).item()
				self.writer.add_scalars('Loss/Value Loss',self.value_loss_dual,episode)
				self.writer.add_scalars('Gradient Normalization/Grad Norm Value',self.grad_norm_value_dual,episode)
				# self.writer.add_scalars('Weights_Critic/Entropy', self.critic_weights_entropy, episode)
				self.writer.add_scalar('Loss/Entropy loss',entropy.item(),episode)
				self.writer.add_scalar('Loss/Policy Loss',policy_loss.item(),episode)
				self.writer.add_scalar('Gradient Normalization/Grad Norm Policy',grad_norm_policy,episode)
			else:
				self.writer.add_scalar('Loss/Entropy loss',entropy.item(),episode)
				self.writer.add_scalar('Loss/Value Loss',value_loss.item(),episode)
				self.writer.add_scalar('Loss/Policy Loss',policy_loss.item(),episode)
				self.writer.add_scalar('Gradient Normalization/Grad Norm Value',grad_norm_value,episode)
				self.writer.add_scalar('Gradient Normalization/Grad Norm Policy',grad_norm_policy,episode)

				# self.calculate_indiv_weights(weights)
				# for i in range(self.num_agents):
				# 	agent_name = 'agent %d' % i
				# 	self.writer.add_scalars('Weights_Critic/Average_Weights/'+agent_name,self.weight_dictionary[agent_name],episode)

				# self.calculate_indiv_weights(weight_policy)
				# for i in range(self.num_agents):
				# 	agent_name = 'agent %d' % i
				# 	self.writer.add_scalars('Weights_Policy/Average_Weights/'+agent_name,self.weight_dictionary[agent_name],episode)
				
				# ENTROPY OF WEIGHTS
				entropy_weights = -torch.mean(torch.sum(weights * torch.log(torch.clamp(weights, 1e-10,1.0)), dim=2))
				self.writer.add_scalar('Weights_Critic/Entropy', entropy_weights.item(), episode)

				# entropy_weights = -torch.mean(torch.sum(weight_policy * torch.log(torch.clamp(weight_policy, 1e-10,1.0)), dim=2))
				# self.writer.add_scalar('Weights_Policy/Entropy', entropy_weights.item(), episode)


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




	def run(self):  
		for episode in range(1,self.max_episodes+1):

			states = self.env.reset()

			images = []

			states_critic,states_actor = self.split_states(states)

			trajectory = []
			episode_reward = 0
			for step in range(1, self.max_time_steps+1):

				if self.gif:
					# At each step, append an image to list
					if not(episode%self.gif_checkpoint):
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

				if self.learn:
					if all(dones) or step == self.max_time_steps:

						trajectory.append([states_critic,next_states_critic,one_hot_actions,one_hot_next_actions,actions,states_actor,next_states_actor,rewards,dones])
						print("*"*100)
						print("EPISODE: {} | REWARD: {} | TIME TAKEN: {} / {} \n".format(episode,np.round(episode_reward,decimals=4),step,self.max_time_steps))
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


			if not(episode%1000) and episode!=0 and self.save:
				if self.critic_type == "ALL":
					torch.save(self.agents.critic_network_1.state_dict(), self.critic_model_path+'_epsiode'+str(episode)+'_MLP_CRITIC_STATE.pt')
					torch.save(self.agents.critic_network_2.state_dict(), self.critic_model_path+'_epsiode'+str(episode)+'_MLP_CRITIC_STATE_ACTION.pt')
					torch.save(self.agents.critic_network_3.state_dict(), self.critic_model_path+'_epsiode'+str(episode)+'_GNN_CRITIC_STATE.pt')
					torch.save(self.agents.critic_network_4.state_dict(), self.critic_model_path+'_epsiode'+str(episode)+'_GNN_CRITIC_STATE_ACTION.pt')
				elif self.critic_type == "ALL_W_POL":
					torch.save(self.agents.critic_network_1.state_dict(), self.critic_model_path+'_epsiode'+str(episode)+'_MLP_CRITIC_STATE.pt')
					torch.save(self.agents.critic_network_2.state_dict(), self.critic_model_path+'_epsiode'+str(episode)+'_MLP_CRITIC_STATE_ACTION.pt')
					torch.save(self.agents.critic_network_3.state_dict(), self.critic_model_path+'_epsiode'+str(episode)+'_GNN_CRITIC_STATE.pt')
					torch.save(self.agents.critic_network_4.state_dict(), self.critic_model_path+'_epsiode'+str(episode)+'_GNN_CRITIC_STATE_ACTION.pt')
					torch.save(self.agents.policy_network.state_dict(), self.actor_model_path+'_epsiode'+str(episode)+'_'+str(self.critic_type)+'.pt')
				elif self.critic_type == "MLPToGNN":
					torch.save(self.agents.critic_network_1.state_dict(), self.critic_model_path+'_epsiode'+str(episode)+'_GNNToMLPV1.pt')
					torch.save(self.agents.critic_network_2.state_dict(), self.critic_model_path+'_epsiode'+str(episode)+'_GNNToMLPV2.pt')
					torch.save(self.agents.critic_network_3.state_dict(), self.critic_model_path+'_epsiode'+str(episode)+'_GNNToMLPV3.pt')
					torch.save(self.agents.critic_network_4.state_dict(), self.critic_model_path+'_epsiode'+str(episode)+'_GNNToMLPV4.pt')
					torch.save(self.agents.critic_network_5.state_dict(), self.critic_model_path+'_epsiode'+str(episode)+'_GNNToMLPV5.pt')
					torch.save(self.agents.critic_network_6.state_dict(), self.critic_model_path+'_epsiode'+str(episode)+'_GNNToMLPV6.pt')
					torch.save(self.agents.critic_network_7.state_dict(), self.critic_model_path+'_epsiode'+str(episode)+'_GNNToMLPV7.pt')
					torch.save(self.agents.critic_network_8.state_dict(), self.critic_model_path+'_epsiode'+str(episode)+'_GNNToMLPV8.pt')
					torch.save(self.agents.policy_network.state_dict(), self.actor_model_path+'_epsiode'+str(episode)+'_'+str(self.critic_type)+'.pt')
				elif "Dual" in self.critic_type:
					torch.save(self.agents.critic_network_1.state_dict(), self.critic_model_path+'_epsiode'+str(episode)+'_'+self.critic_type+'_.pt')
					torch.save(self.agents.critic_network_2.state_dict(), self.critic_model_path+'_epsiode'+str(episode)+'_'+self.critic_type+'_.pt')
					torch.save(self.agents.policy_network.state_dict(), self.actor_model_path+'_epsiode'+str(episode)+'_'+str(self.critic_type)+'.pt')  
				else:	
					torch.save(self.agents.critic_network.state_dict(), self.critic_model_path+'_epsiode'+str(episode)+'_'+str(self.critic_type)+'.pt')
					torch.save(self.agents.policy_network.state_dict(), self.actor_model_path+'_epsiode'+str(episode)+'_'+str(self.critic_type)+'.pt')  

			if self.learn:
				self.update(trajectory,episode) 
			elif self.gif and not(episode%self.gif_checkpoint):
				print("GENERATING GIF")
				self.make_gif(np.array(images),self.gif_path)