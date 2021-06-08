import os
import torch
import torch.nn.functional as F 
import torch.optim as optim
from torch.distributions import Categorical
import torch.autograd as autograd
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from a2c_agent import A2CAgent
import datetime



class MAA2C:

	def __init__(self,env, dictionary):
		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		# self.device = "cpu"
		self.env_name = dictionary["env"]
		self.env = env
		self.gif = dictionary["gif"]
		self.save = dictionary["save"]
		self.learn = dictionary["learn"]
		self.num_agents = env.n
		self.num_actions = self.env.action_space[0].n
		self.date_time = f"{datetime.datetime.now():%d-%m-%Y}"

		self.max_episodes = dictionary["max_episodes"]
		self.max_time_steps = dictionary["max_time_steps"]

		self.weight_dictionary = {}

		# SAVE REWARDS 
		self.rewards = []
		self.rewards_mean_per_1000_eps = []
		self.timesteps = []
		self.timesteps_mean_per_1000_eps = []
		self.collision_rates = []
		self.collison_rate_mean_per_1000_eps = []

		for i in range(self.num_agents):
			agent_name = 'agent %d' % i
			self.weight_dictionary[agent_name] = {}
			for j in range(self.num_agents):
				agent_name_ = 'agent %d' % j
				self.weight_dictionary[agent_name][agent_name_] = 0


		if self.env_name == "crowd_nav":
			self.num_agents = dictionary["num_agents"]
			self.num_people = dictionary["num_people"]
		elif self.env_name == "predator_prey":
			self.num_predators = dictionary["num_predators"]
			self.num_preys = dictionary["num_preys"]

		if dictionary["one_policy_per_agent"]:
			if self.env_name in ["predator_prey"]:
				self.num_policy_net = self.num_predators
			else:
				self.num_policy_net = self.num_agents


		self.agents = A2CAgent(self.env, dictionary)


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

			self.policy_eval_dir = dictionary["policy_eval_dir"]
			try: 
				os.makedirs(self.policy_eval_dir, exist_ok = True) 
				print("Policy Eval Directory created successfully") 
			except OSError as error: 
				print("Policy Eval Directory can not be created")


			# paths for models, tensorboard and gifs
			self.critic_model_path = critic_dir+str(self.date_time)+'VN_ATN_FCN_lr'+str(self.agents.value_lr)+'_PN_ATN_FCN_lr'+str(self.agents.policy_lr)+'_GradNorm0.5_Entropy'+str(self.agents.entropy_pen)+'_trace_decay'+str(self.agents.trace_decay)+"topK_"+str(self.agents.top_k)+"select_above_threshold"+str(self.agents.select_above_threshold)+"softmax_cut_threshold"+str(self.agents.softmax_cut_threshold)+"_tau"+str(self.agents.tau)+"_select_above_threshold"+str(self.agents.select_above_threshold)+"_tdlambda"+str(self.agents.td_lambda)+"_l1pen"+str(self.agents.l1_pen)+str(self.agents.critic_update_type)+str(self.agents.critic_loss_type)
			self.actor_model_path = actor_dir+str(self.date_time)+'_PN_ATN_FCN_lr'+str(self.agents.policy_lr)+'VN_SAT_FCN_lr'+str(self.agents.value_lr)+'_GradNorm0.5_Entropy'+str(self.agents.entropy_pen)+'_trace_decay'+str(self.agents.trace_decay)+"topK_"+str(self.agents.top_k)+"select_above_threshold"+str(self.agents.select_above_threshold)+"softmax_cut_threshold"+str(self.agents.softmax_cut_threshold)+"_tau"+str(self.agents.tau)+"_select_above_threshold"+str(self.agents.select_above_threshold)+"_tdlambda"+str(self.agents.td_lambda)+"_l1pen"+str(self.agents.l1_pen)+str(self.agents.critic_update_type)+str(self.agents.critic_loss_type)
			self.tensorboard_path = tensorboard_dir+str(self.date_time)+'VN_SAT_FCN_lr'+str(self.agents.value_lr)+'_PN_ATN_FCN_lr'+str(self.agents.policy_lr)+'_GradNorm0.5_Entropy'+str(self.agents.entropy_pen)+'_trace_decay'+str(self.agents.trace_decay)+"topK_"+str(self.agents.top_k)+"select_above_threshold"+str(self.agents.select_above_threshold)+"softmax_cut_threshold"+str(self.agents.softmax_cut_threshold)+"_tau"+str(self.agents.tau)+"_select_above_threshold"+str(self.agents.select_above_threshold)+"_tdlambda"+str(self.agents.td_lambda)+"_l1pen"+str(self.agents.l1_pen)+str(self.agents.critic_update_type)+str(self.agents.critic_loss_type)
			self.writer = SummaryWriter(self.tensorboard_path)

		if self.gif:
			gif_dir = dictionary["gif_dir"]
			try: 
				os.makedirs(gif_dir, exist_ok = True) 
				print("Gif Directory created successfully") 
			except OSError as error: 
				print("Gif Directory can not be created")
			self.gif_path = gif_dir+str(self.date_time)+'VN_SAT_FCN_lr'+str(self.agents.value_lr)+'_PN_ATN_FCN_lr'+str(self.agents.policy_lr)+'_GradNorm0.5_Entropy'+str(self.agents.entropy_pen)+"topK_"+str(self.agents.top_k)+"select_above_threshold"+str(self.agents.select_above_threshold)+"softmax_cut_threshold"+str(self.agents.softmax_cut_threshold)




	def get_actions(self,states, states_other=None):
		if self.learn:
			if self.env_name in ["paired_by_sharing_goals", "collision_avoidance", "multi_circular"]:
				actions = self.agents.get_action(states)
			elif self.env_name in ["crowd_nav", "predator_prey"]:
				actions = self.agents.get_action(states, states_other)
		else:
			with torch.no_grad():
				if self.env_name in ["paired_by_sharing_goals", "collision_avoidance", "multi_circular"]:
					actions = self.agents.get_action(states)
				elif self.env_name in ["crowd_nav", "predator_prey"]:
					actions = self.agents.get_action(states, states_other)
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
		
		if self.env_name in ["paired_by_sharing_goals", "collision_avoidance", "multi_circular"]:
			value_loss,policy_loss,entropy,grad_norm_value,grad_norm_policy,weights,weight_policy = self.agents.update(states_critic,next_states_critic,one_hot_actions,one_hot_next_actions,actions,states_actor,next_states_actor,rewards,dones)
		elif self.env_name in ["crowd_nav"]:
			states_critic_people = torch.FloatTensor([sars[9] for sars in trajectory]).to(self.device)
			next_states_critic_people = torch.FloatTensor([sars[10] for sars in trajectory]).to(self.device)

			one_hot_actions_people = torch.FloatTensor([sars[11] for sars in trajectory]).to(self.device)
			one_hot_next_actions_people = torch.FloatTensor([sars[12] for sars in trajectory]).to(self.device)

			states_actor_people = torch.FloatTensor([sars[13] for sars in trajectory]).to(self.device)
			next_states_actor_people = torch.FloatTensor([sars[14] for sars in trajectory]).to(self.device)

			value_loss,policy_loss,entropy,grad_norm_value,grad_norm_policy,weights, weight_policy, weights_people, weight_policy_people = self.agents.update(states_critic,next_states_critic,one_hot_actions,one_hot_next_actions,actions,states_actor,next_states_actor,rewards,dones,states_critic_people,next_states_critic_people,one_hot_actions_people,one_hot_next_actions_people,states_actor_people,next_states_actor_people)



		if self.save:
			self.writer.add_scalar('Loss/Entropy loss',entropy.item(),episode)
			self.writer.add_scalar('Loss/Value Loss',value_loss.item(),episode)
			self.writer.add_scalar('Loss/Policy Loss',policy_loss.item(),episode)
			self.writer.add_scalar('Gradient Normalization/Grad Norm Value',grad_norm_value,episode)
			self.writer.add_scalar('Gradient Normalization/Grad Norm Policy',grad_norm_policy,episode)

			# if self.env_name in ["collision_avoidance", "multi_circular"]:
				# # INDIVIDUAL WEIGHTS
				# self.calculate_indiv_weights(weights)
				# for i in range(self.num_agents):
				# 	agent_name = 'agent %d' % i
				# 	self.writer.add_scalars('Weights/Average_Weights/'+agent_name,self.weight_dictionary[agent_name],episode)
			# elif self.env_name == "paired_by_sharing_goals":
				# # PAIRED AGENT WEIGHTS
				# paired_agent_avg_weight, unpaired_agent_avg_weight = self.calculate_weights(weights)
				# self.writer.add_scalars('Weights/Average_Weights',{'Paired':paired_agent_avg_weight,'Unpaired':unpaired_agent_avg_weight},episode)

			# ENTROPY OF WEIGHTS
			entropy_weights = -torch.mean(torch.sum(weights * torch.log(torch.clamp(weights, 1e-10,1.0)), dim=2))
			self.writer.add_scalar('Weights/Entropy', entropy_weights.item(), episode)

			entropy_weights = -torch.mean(torch.sum(weight_policy * torch.log(torch.clamp(weight_policy, 1e-10,1.0)), dim=2))
			self.writer.add_scalar('Weights_Policy/Entropy', entropy_weights.item(), episode)

			if self.env_name in ["crowd_nav"]:
				entropy_weights = -torch.mean(torch.sum(weights_people * torch.log(torch.clamp(weights_people, 1e-10,1.0)), dim=2))
				self.writer.add_scalar('Weights/Entropy_People', entropy_weights.item(), episode)

				entropy_weights = -torch.mean(torch.sum(weight_policy_people * torch.log(torch.clamp(weight_policy_people, 1e-10,1.0)), dim=2))
				self.writer.add_scalar('Weights_Policy/Entropy_People', entropy_weights.item(), episode)


	def split_states(self,states,for_who):

		states_critic = []
		states_actor = []

		if for_who == "agents":
			for i in range(self.num_agents):
				states_critic.append(states[i][0])
				states_actor.append(states[i][1])
		elif for_who == "people":
			for i in range(self.num_people):
				states_critic.append(states[i][0])
				states_actor.append(states[i][1])
		elif for_who == "predator":
			for i in range(self.num_predators):
				states_critic.append(states[i][0])
				states_actor.append(states[i][1])
		elif for_who == "prey":
			for i in range(self.num_preys):
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


	def get_prey_actions(self, states_prey, states_predator, random = False):
		prey_speed = 1.3
		dt = 0.1
		prey_actions = []

		left = np.array([-1,0])
		right = np.array([1,0])
		up = np.array([0,1])
		down = np.array([0,-1])

		no_act = 0
		left_act = 1
		right_act = 2
		down_act = 3
		up_act = 4

		for prey_x_y in states_prey[:,0:2]:
			dist_from_predator = [np.sqrt(np.sum(np.square(prey_x_y - predator_x_y))) for predator_x_y in states_predator[:,0:2]]
			min_dist_predator = min(dist_from_predator)
			min_dist_predator_index = np.argmin(dist_from_predator)

			if random:
				valid_action = [no_act]

				# taking left
				if prey_x_y[0] - prey_speed*dt >= -1 and np.sqrt(np.sum(np.square(prey_x_y + prey_speed*left*dt - states_predator[min_dist_predator_index,0:2]))) > min_dist_predator:
					valid_action.append(left_act)
				# taking right
				if prey_x_y[0] + prey_speed*dt <= 1 and np.sqrt(np.sum(np.square(prey_x_y + prey_speed*left*dt - states_predator[min_dist_predator_index,0:2]))) > min_dist_predator:
					valid_action.append(right_act)
				# taking up
				if prey_x_y[1] + prey_speed*dt <= 1 and np.sqrt(np.sum(np.square(prey_x_y + prey_speed*left*dt - states_predator[min_dist_predator_index,0:2]))) > min_dist_predator:
					valid_action.append(up_act)
				# taking down
				if prey_x_y[1] - prey_speed*dt >= -1 and np.sqrt(np.sum(np.square(prey_x_y + prey_speed*left*dt - states_predator[min_dist_predator_index,0:2]))) > min_dist_predator:
					valid_action.append(down_act)

				prey_actions.append(random.choice(valid_action))

			else:
				valid_action = [no_act]
				distance_metric = [min_dist_predator]
				if prey_x_y[0] - prey_speed*dt >= -1 and np.sqrt(np.sum(np.square(prey_x_y + prey_speed*left*dt - states_predator[min_dist_predator_index,0:2]))) > min_dist_predator:
					valid_action.append(left_act)
					distance_metric.append(np.sqrt(np.sum(np.square(prey_x_y + prey_speed*left*dt - states_predator[min_dist_predator_index,0:2]))))
				# taking right
				if prey_x_y[0] + prey_speed*dt <= 1 and np.sqrt(np.sum(np.square(prey_x_y + prey_speed*left*dt - states_predator[min_dist_predator_index,0:2]))) > min_dist_predator:
					valid_action.append(right_act)
					distance_metric.append(np.sqrt(np.sum(np.square(prey_x_y + prey_speed*left*dt - states_predator[min_dist_predator_index,0:2]))))
				# taking up
				if prey_x_y[1] + prey_speed*dt <= 1 and np.sqrt(np.sum(np.square(prey_x_y + prey_speed*left*dt - states_predator[min_dist_predator_index,0:2]))) > min_dist_predator:
					valid_action.append(up_act)
					distance_metric.append(np.sqrt(np.sum(np.square(prey_x_y + prey_speed*left*dt - states_predator[min_dist_predator_index,0:2]))))
				# taking down
				if prey_x_y[1] - prey_speed*dt >= -1 and np.sqrt(np.sum(np.square(prey_x_y + prey_speed*left*dt - states_predator[min_dist_predator_index,0:2]))) > min_dist_predator:
					valid_action.append(down_act)
					distance_metric.append(np.sqrt(np.sum(np.square(prey_x_y + prey_speed*left*dt - states_predator[min_dist_predator_index,0:2]))))


				longest_dist_ind = np.argmax(distance_metric)

				prey_actions.append(valid_action[longest_dist_ind])


		return prey_actions




	def run(self):  
		self.rewards = []
		self.rewards_mean_per_1000_eps = []
		self.timesteps = []
		self.timesteps_mean_per_1000_eps = []
		self.collision_rates = []
		self.collison_rate_mean_per_1000_eps = []
		for episode in range(1,self.max_episodes+1):

			# RANDOMIZING NUMBER OF AGENTS
			# self.num_agents = self.env.n
			# self.agents.num_agents = self.num_agents
			# self.agents.critic_network.num_agents = self.num_agents
			# self.agents.policy_network.num_agents = self.num_agents
			# self.agents.get_scaling_factor()

			states = self.env.reset()

			if self.env_name in ["crowd_nav"]:
				states_agents = states[:self.num_agents]
				states_people = states[self.num_agents:]
				states_critic,states_actor = self.split_states(states_agents, "agents")
				states_critic_people, states_actor_people = self.split_states(states_people, "people")
			elif self.env_name in ["predator_prey"]:
				states_predator = states[:self.num_predators]
				states_prey = states[self.num_predators:]
				states_critic,states_actor = self.split_states(states_predator, "predator")
				states_critic_prey, states_actor_prey = self.split_states(states_prey, "prey")
			else:
				states_critic,states_actor = self.split_states(states, "agents")

			if self.gif:
				images = []
				gif_checkpoint = 10

			trajectory = []
			episode_reward = 0
			episode_collision_rate = 0
			for step in range(1,self.max_time_steps+1):

				if self.gif and not(episode%gif_checkpoint):
					# At each step, append an image to list
					images.append(np.squeeze(self.env.render(mode='rgb_array')))

				if self.env_name in ["paired_by_sharing_goals", "collision_avoidance", "multi_circular"]:
					actions = self.get_actions(states_actor)
				elif self.env_name in ["crowd_nav"]:
					actions = self.get_actions(states_actor, states_actor_people)
				elif self.env_name in ["predator_prey"]:
					actions = self.get_actions(states_actor, states_actor_prey)

				one_hot_actions = np.zeros((self.num_agents,self.num_actions))
				for i,act in enumerate(actions):
					one_hot_actions[i][act] = 1


				if self.env_name in ["paired_by_sharing_goals", "collision_avoidance", "multi_circular"]:
					next_states,rewards,dones,info = self.env.step(actions)
					next_states_critic,next_states_actor = self.split_states(next_states, "agents")
					# next actions
					next_actions = self.get_actions(next_states_actor)

					one_hot_next_actions = np.zeros((self.num_agents,self.num_actions))
					for i,act in enumerate(next_actions):
						one_hot_next_actions[i][act] = 1

				elif self.env_name in ["crowd_nav"]:
					actions_people = []
					for i in range(self.num_people):
						if i%4 == 0:
							actions_people.append(2)
						elif i%4 == 1:
							actions_people.append(4)
						elif i%4 == 2:
							actions_people.append(1)
						elif i%4 == 3:
							actions_people.append(3)
					one_hot_actions_people = np.zeros((self.num_people,self.num_actions))
					for i,act in enumerate(actions_people):
						one_hot_actions_people[i][act] = 1

					next_states,rewards,dones,info = self.env.step(actions+actions_people)
					next_states_agents = next_states[:self.num_agents]
					next_states_people = next_states[self.num_agents:]
					next_states_critic,next_states_actor = self.split_states(next_states_agents, "agents")
					next_states_critic_people,next_states_actor_people = self.split_states(next_states_people, "people")


					next_actions = self.get_actions(next_states_actor, next_states_actor_people)
					one_hot_next_actions = np.zeros((self.num_agents,self.num_actions))
					for i,act in enumerate(next_actions):
						one_hot_next_actions[i][act] = 1

					next_actions_people = []
					for i in range(self.num_people):
						if i%4 == 0:
							next_actions_people.append(2)
						elif i%4 == 1:
							next_actions_people.append(4)
						elif i%4 == 2:
							next_actions_people.append(1)
						elif i%4 == 3:
							next_actions_people.append(3)
					one_hot_next_actions_people = np.zeros((self.num_people,self.num_actions))
					for i,act in enumerate(next_actions_people):
						one_hot_next_actions[i][act] = 1

					rewards = rewards[:self.num_agents]
					dones = dones[:self.num_agents]

				elif self.env_name in ["predator_prey"]:
					actions_prey = self.get_prey_actions(states_critic_prey, states_critic)

					one_hot_actions_prey = np.zeros((self.num_preys,self.num_actions))
					for i,act in enumerate(actions_prey):
						one_hot_actions_prey[i][act] = 1

					next_states,rewards,dones,info = self.env.step(actions+actions_prey)
					next_states_agents = next_states[:self.num_predators]
					next_states_prey = next_states[self.num_predators:]
					next_states_critic,next_states_actor = self.split_states(next_states_agents, "predators")
					next_states_critic_prey,next_states_actor_prey = self.split_states(next_states_prey, "prey")


					next_actions = self.get_actions(next_states_actor, next_states_actor_prey)
					one_hot_next_actions = np.zeros((self.num_predators,self.num_actions))
					for i,act in enumerate(next_actions):
						one_hot_next_actions[i][act] = 1

					next_actions_prey = self.get_prey_actions(next_states_critic_prey, next_states_critic)
					one_hot_next_actions_prey = np.zeros((self.num_preys,self.num_actions))
					for i,act in enumerate(next_actions_prey):
						one_hot_next_actions_prey[i][act] = 1

					rewards = rewards[:self.num_predators]
					dones = dones[:self.predators]
				
				if self.env_name in ["collision_avoidance", "multi_circular"]:
					collision_rate = [value[1] for value in rewards]
					rewards = [value[0] for value in rewards]
				episode_reward += np.sum(rewards)
				episode_collision_rate += np.sum(collision_rate)

				if all(dones) or step == self.max_time_steps:
					print("*"*100)
					print("EPISODE: {} | REWARD: {} | TIME TAKEN: {} / {} \n".format(episode,np.round(episode_reward,decimals=4),step,self.max_time_steps))
					print("*"*100)
					final_timestep = step

					if self.save:
						self.writer.add_scalar('Reward Incurred/Length of the episode',step,episode)
						self.writer.add_scalar('Reward Incurred/Reward',episode_reward,episode)
						if self.env_name in ["collision_avoidance", "multi_circular"]:
							self.writer.add_scalar('Reward Incurred/Collison Rate',episode_collision_rate,episode)

					break


				if self.learn:
					if self.env_name in ["paired_by_sharing_goals", "collision_avoidance", "multi_circular"]:
						trajectory.append([states_critic,next_states_critic,one_hot_actions,one_hot_next_actions,actions,states_actor,next_states_actor,rewards,dones])
					elif self.env_name in ["crowd_nav"]:
						trajectory.append([states_critic,next_states_critic,one_hot_actions,one_hot_next_actions,actions,states_actor,next_states_actor,rewards,dones,states_critic_people,next_states_critic_people,one_hot_actions_people,one_hot_next_actions_people,states_actor_people,next_states_actor_people])


				states_critic,states_actor = next_states_critic,next_states_actor
				states = next_states


			self.rewards.append(episode_reward)
			self.timesteps.append(final_timestep)
			self.collision_rates.append(episode_collision_rate)
			if episode > 1000 and episode%1000:
				self.rewards_mean_per_1000_eps.append(sum(self.rewards[episode-1000:episode])/1000)
				self.timesteps_mean_per_1000_eps.append(sum(self.timesteps[episode-1000:episode])/1000)
				if self.env_name in ["collision_avoidance", "multi_circular"]:
					self.collison_rate_mean_per_1000_eps.append(sum(self.collision_rates[episode-1000:episode])/1000)

			if not(episode%1000) and self.save:
				torch.save(self.agents.critic_network.state_dict(), self.critic_model_path+'_epsiode'+str(episode)+'.pt')
				torch.save(self.agents.policy_network.state_dict(), self.actor_model_path+'_epsiode'+str(episode)+'.pt')  

			if self.learn:
				self.update(trajectory,episode) 
			if self.gif and not(episode%gif_checkpoint):
				print("GENERATING GIF")
				self.make_gif(np.array(images),self.gif_path+"_episode"+str(episode)+'.gif')

		np.save(os.path.join(self.policy_eval_dir,self.env_name+"reward_list"), np.array(self.rewards), allow_pickle=True, fix_imports=True)
		np.save(os.path.join(self.policy_eval_dir,self.env_name+"mean_rewards_per_1000_eps"), np.array(self.rewards_mean_per_1000_eps), allow_pickle=True, fix_imports=True)
		np.save(os.path.join(self.policy_eval_dir,self.env_name+"timestep_list"), np.array(self.timesteps), allow_pickle=True, fix_imports=True)
		np.save(os.path.join(self.policy_eval_dir,self.env_name+"mean_timestep_per_1000_eps"), np.array(self.timesteps_mean_per_1000_eps), allow_pickle=True, fix_imports=True)
		if self.env_name in ["collision_avoidance", "multi_circular"]:
			np.save(os.path.join(self.policy_eval_dir,self.env_name+"collision_rate_list"), np.array(self.collision_rates), allow_pickle=True, fix_imports=True)
			np.save(os.path.join(self.policy_eval_dir,self.env_name+"mean_collision_rate_per_1000_eps"), np.array(self.collison_rate_mean_per_1000_eps), allow_pickle=True, fix_imports=True)