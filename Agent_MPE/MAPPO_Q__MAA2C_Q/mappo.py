import os
import time
from comet_ml import Experiment
import numpy as np
from ppo_agent import PPOAgent
import torch
import datetime
from torch.distributions import Categorical



class MAPPO:

	def __init__(self, env, dictionary):
		if dictionary["device"] == "gpu":
			self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		else:
			self.device = "cpu"
		self.env = env
		self.gif = dictionary["gif"]
		self.save_model = dictionary["save_model"]
		self.save_model_checkpoint = dictionary["save_model_checkpoint"]
		self.save_comet_ml_plot = dictionary["save_comet_ml_plot"]
		self.learn = dictionary["learn"]
		self.gif_checkpoint = dictionary["gif_checkpoint"]
		self.eval_policy = dictionary["eval_policy"]
		self.num_agents = self.env.n
		self.num_actions = self.env.action_space[0].n
		self.date_time = f"{datetime.datetime.now():%d-%m-%Y}"
		self.env_name = dictionary["env"]
		self.test_num = dictionary["test_num"]
		self.max_episodes = dictionary["max_episodes"]
		self.max_time_steps = dictionary["max_time_steps"]
		self.experiment_type = dictionary["experiment_type"]
		self.update_ppo_agent = dictionary["update_ppo_agent"]
		self.model_path_value = dictionary["model_path_value"]
		self.model_path_policy = dictionary["model_path_policy"]
		self.update_type = dictionary["update_type"]


		self.comet_ml = None
		if self.save_comet_ml_plot:
			self.comet_ml = Experiment("im5zK8gFkz6j07uflhc3hXk8I",project_name=dictionary["test_num"])
			self.comet_ml.log_parameters(dictionary)


		self.agents = PPOAgent(self.env, dictionary, self.comet_ml)

		if self.save_model:
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

			
			self.critic_model_path = critic_dir+"critic"
			self.actor_model_path = actor_dir+"actor"
			

		if self.gif:
			gif_dir = dictionary["gif_dir"]
			try: 
				os.makedirs(gif_dir, exist_ok = True) 
				print("Gif Directory created successfully") 
			except OSError as error: 
				print("Gif Directory can not be created")
			self.gif_path = gif_dir+self.env_name+'.gif'


		if self.eval_policy:
			self.policy_eval_dir = dictionary["policy_eval_dir"]
			try: 
				os.makedirs(self.policy_eval_dir, exist_ok = True) 
				print("Policy Eval Directory created successfully") 
			except OSError as error: 
				print("Policy Eval Directory can not be created")

		self.environment_time = 0.0

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
		if self.eval_policy:
			self.rewards = []
			self.rewards_mean_per_1000_eps = []
			self.timesteps = []
			self.timesteps_mean_per_1000_eps = []
			self.collision_rates = []
			self.collison_rate_mean_per_1000_eps = []

		for episode in range(1,self.max_episodes+1):

			states = self.env.reset()

			images = []

			states_critic, states_actor = self.split_states(states)

			trajectory = []
			episode_reward = 0
			episode_collision_rate = 0
			episode_goal_reached = 0
			final_timestep = self.max_time_steps
			for step in range(1, self.max_time_steps+1):

				if self.gif:
					# At each step, append an image to list
					if not(episode%self.gif_checkpoint):
						images.append(np.squeeze(self.env.render(mode='rgb_array')))
					# Advance a step and render a new image
					with torch.no_grad():
						actions = self.agents.get_action(states_actor, greedy=True)
				else:
					actions = self.agents.get_action(states_actor)

				one_hot_actions = np.zeros((self.num_agents,self.num_actions))
				for i,act in enumerate(actions):
					one_hot_actions[i][act] = 1

				start_env_time = time.process_time()

				next_states, rewards, dones, info = self.env.step(actions)
				
				end_env_time = time.process_time()
				self.environment_time += end_env_time - start_env_time

				next_states_critic, next_states_actor = self.split_states(next_states)

				if "crossing" in self.env_name:
					collision_rate = [value[1] for value in rewards]
					goal_reached = [value[2] for value in rewards]
					rewards = [value[0] for value in rewards]
					episode_collision_rate += np.sum(collision_rate)
					episode_goal_reached += np.sum(goal_reached)


				# if step == self.max_time_steps:
				# 	dones = [True for _ in range(self.num_agents)]

				self.agents.buffer.states_critic.append(states_critic)
				self.agents.buffer.states_actor.append(states_actor)
				self.agents.buffer.actions.append(actions)
				self.agents.buffer.one_hot_actions.append(one_hot_actions)
				self.agents.buffer.dones.append(dones)
				self.agents.buffer.rewards.append(rewards)

				episode_reward += np.sum(rewards)

				states_critic, states_actor = next_states_critic, next_states_actor
				states = next_states

				if all(dones) or step == self.max_time_steps:
					print("*"*100)
					print("EPISODE: {} | REWARD: {} | TIME TAKEN: {} / {} \n".format(episode,np.round(episode_reward,decimals=4),step,self.max_time_steps))
					print("*"*100)

					final_timestep = step

					if self.save_comet_ml_plot:
						self.comet_ml.log_metric('Episode_Length', step, episode)
						self.comet_ml.log_metric('Reward', episode_reward, episode)
						if "crossing" in self.env_name:
							self.comet_ml.log_metric('Number of Collision', episode_collision_rate, episode)
							self.comet_ml.log_metric('Num Agents Goal Reached', np.sum(dones), episode)

					break

			if self.eval_policy:
				self.rewards.append(episode_reward)
				self.timesteps.append(final_timestep)
				if "crossing" in self.env_name:
					self.collision_rates.append(episode_collision_rate)

			if episode > self.save_model_checkpoint and self.eval_policy:
				self.rewards_mean_per_1000_eps.append(sum(self.rewards[episode-self.save_model_checkpoint:episode])/self.save_model_checkpoint)
				self.timesteps_mean_per_1000_eps.append(sum(self.timesteps[episode-self.save_model_checkpoint:episode])/self.save_model_checkpoint)
				if "crossing" in self.env_name:
					self.collison_rate_mean_per_1000_eps.append(sum(self.collision_rates[episode-self.save_model_checkpoint:episode])/self.save_model_checkpoint)


			if not(episode%self.save_model_checkpoint) and episode!=0 and self.save_model:	
				torch.save(self.agents.critic_network.state_dict(), self.critic_model_path+'_epsiode'+str(episode)+'.pt')
				torch.save(self.agents.policy_network.state_dict(), self.actor_model_path+'_epsiode'+str(episode)+'.pt')  

			if self.learn and not(episode%self.update_ppo_agent) and episode != 0:
				if self.update_type == "ppo":
					self.agents.update(episode) 
				if self.update_type == "ppo_V":
					self.agents.V_update(episode) 
				elif self.update_type == "a2c":
					self.agents.a2c_update(episode)
			elif self.gif and not(episode%self.gif_checkpoint):
				print("GENERATING GIF")
				self.make_gif(np.array(images),self.gif_path)


			if self.eval_policy and not(episode%self.save_model_checkpoint) and episode!=0:
				np.save(os.path.join(self.policy_eval_dir,self.test_num+"reward_list"), np.array(self.rewards), allow_pickle=True, fix_imports=True)
				np.save(os.path.join(self.policy_eval_dir,self.test_num+"mean_rewards_per_1000_eps"), np.array(self.rewards_mean_per_1000_eps), allow_pickle=True, fix_imports=True)
				np.save(os.path.join(self.policy_eval_dir,self.test_num+"timestep_list"), np.array(self.timesteps), allow_pickle=True, fix_imports=True)
				np.save(os.path.join(self.policy_eval_dir,self.test_num+"mean_timestep_per_1000_eps"), np.array(self.timesteps_mean_per_1000_eps), allow_pickle=True, fix_imports=True)
				if "crossing" in self.env_name:
					np.save(os.path.join(self.policy_eval_dir,self.test_num+"collision_rate_list"), np.array(self.collision_rates), allow_pickle=True, fix_imports=True)
					np.save(os.path.join(self.policy_eval_dir,self.test_num+"mean_collision_rate_per_1000_eps"), np.array(self.collison_rate_mean_per_1000_eps), allow_pickle=True, fix_imports=True)

				if "prd" in self.experiment_type and "crossing" in self.env_name:
					np.save(os.path.join(self.policy_eval_dir,self.test_num+"num_relevant_agents_in_relevant_set"), np.array(self.agents.num_relevant_agents_in_relevant_set), allow_pickle=True, fix_imports=True)
					np.save(os.path.join(self.policy_eval_dir,self.test_num+"num_non_relevant_agents_in_relevant_set"), np.array(self.agents.num_non_relevant_agents_in_relevant_set), allow_pickle=True, fix_imports=True)
					np.save(os.path.join(self.policy_eval_dir,self.test_num+"false_positive_rate"), np.array(self.agents.false_positive_rate), allow_pickle=True, fix_imports=True)

		print("Environment Time:", self.environment_time/(self.max_episodes*self.max_time_steps))
		print("Total Update Time:", (self.agents.update_time+self.agents.forward_time)/self.max_episodes)
		print("Update Time:", self.agents.update_time/self.max_episodes)
		print("Forward Time:", self.agents.forward_time/self.max_episodes)

	def test(self):
		self.reward_data_points = []
		num_data_points = 100
		num_episodes = 100
		num_evals = 100

		try: 
			os.makedirs("../../../tests/Crossing/evaluate/", exist_ok = True) 
			print("Eval Directory created successfully") 
		except OSError as error: 
			print("Eval Directory can not be created")

		for i in range(1, num_data_points+1):

			self.agents.critic_network.load_state_dict(torch.load(self.model_path_value))
			self.agents.policy_network.load_state_dict(torch.load(self.model_path_policy))
			self.agents.policy_network_old.load_state_dict(torch.load(self.model_path_policy))
			# self.agents.critic_network.reset_parameters()
			# self.agents.policy_network.reset_parameters()
			# self.agents.policy_network_old.reset_parameters()

			for episode in range(1, num_episodes+1):
				episode_reward = 0

				states = self.env.reset()
				states_critic, states_actor = self.split_states(states)

				for step in range(1, self.max_time_steps+1):

					actions = self.agents.get_action(states_actor)

					one_hot_actions = np.zeros((self.num_agents,self.num_actions))
					for i,act in enumerate(actions):
						one_hot_actions[i][act] = 1

					next_states, rewards, dones, info = self.env.step(actions)
					next_states_critic, next_states_actor = self.split_states(next_states)
					rewards = [value[0] for value in rewards]

					self.agents.buffer.states_critic.append(states_critic)
					self.agents.buffer.states_actor.append(states_actor)
					self.agents.buffer.actions.append(actions)
					self.agents.buffer.one_hot_actions.append(one_hot_actions)
					self.agents.buffer.dones.append(dones)
					self.agents.buffer.rewards.append(rewards)

					episode_reward += np.sum(rewards)

					states_critic, states_actor = next_states_critic, next_states_actor
					states = next_states
				
					if all(dones) or step == self.max_time_steps:
						final_timestep = step
						print("*"*100)
						print("EPISODE: {} | REWARD: {} | TIME TAKEN: {} / {} \n".format(episode,np.round(episode_reward,decimals=4),final_timestep,self.max_time_steps))
						print("*"*100)
						break

				# print("update")
				self.agents.update(episode)

			eval_reward = 0
			for num_eval in range(1, num_evals+1):
				episode_reward = 0

				states = self.env.reset()
				states_critic, states_actor = self.split_states(states)
				for step in range(1, self.max_time_steps+1):

					with torch.no_grad():
						state_policy = torch.FloatTensor(states_actor).to(self.device)
						dists = self.agents.policy_network(state_policy)
						actions = [Categorical(dist).sample().detach().cpu().item() for dist in dists]

					next_states, rewards, dones, info = self.env.step(actions)
					next_states_critic, next_states_actor = self.split_states(next_states)
					rewards = [value[0] for value in rewards]

					episode_reward += np.sum(rewards)

					states_critic, states_actor = next_states_critic, next_states_actor
					states = next_states
				
					if all(dones) or step == self.max_time_steps:
						eval_reward += episode_reward
						final_timestep = step
						print("*"*100)
						print("EPISODE: {} | REWARD: {} | TIME TAKEN: {} / {} \n".format(num_eval,np.round(episode_reward,decimals=4),final_timestep,self.max_time_steps))
						print("*"*100)
						break

			# print(eval_reward/num_evals)
			self.reward_data_points.append(eval_reward/num_evals)

		print("DATA POINTS")
		print(self.reward_data_points)

		np.save(os.path.join("../../../tests/Crossing/evaluate/"+self.experiment_type+"_1000"), np.array(self.reward_data_points), allow_pickle=True, fix_imports=True)


	def run_gradvar_exp(self, episode):
		self.reward_data_points = []
		num_data_points = 100
		num_episodes = 100
		num_evals = 100

		policy_grads = []

		if torch.cuda.is_available() is False:
			# For CPU
			self.agents.critic_network.load_state_dict(torch.load(self.model_path_value,map_location=torch.device('cpu')))
			self.agents.critic_network_old.load_state_dict(torch.load(self.model_path_value,map_location=torch.device('cpu')))
			self.agents.policy_network.load_state_dict(torch.load(self.model_path_policy,map_location=torch.device('cpu')))
			self.agents.policy_network_old.load_state_dict(torch.load(self.model_path_policy,map_location=torch.device('cpu')))
		else:
			# For GPU
			self.agents.critic_network.load_state_dict(torch.load(self.model_path_value))
			self.agents.critic_network_old.load_state_dict(torch.load(self.model_path_value))
			self.agents.policy_network.load_state_dict(torch.load(self.model_path_policy))
			self.agents.policy_network_old.load_state_dict(torch.load(self.model_path_policy))

		for episode in range(1, self.max_episodes+1):
			episode_reward = 0

			states = self.env.reset()
			states_critic, states_actor = self.split_states(states)

			for step in range(1, self.max_time_steps+1):

				actions = self.agents.get_action(states_actor)

				one_hot_actions = np.zeros((self.num_agents,self.num_actions))
				for i,act in enumerate(actions):
					one_hot_actions[i][act] = 1

				next_states, rewards, dones, info = self.env.step(actions)
				next_states_critic, next_states_actor = self.split_states(next_states)
				rewards = [value[0] for value in rewards]

				self.agents.buffer.states_critic.append(states_critic)
				self.agents.buffer.states_actor.append(states_actor)
				self.agents.buffer.actions.append(actions)
				self.agents.buffer.one_hot_actions.append(one_hot_actions)
				self.agents.buffer.dones.append(dones)
				self.agents.buffer.rewards.append(rewards)

				episode_reward += np.sum(rewards)

				states_critic, states_actor = next_states_critic, next_states_actor
				states = next_states
			
				if all(dones) or step == self.max_time_steps:
					final_timestep = step
					print("*"*100)
					print("EPISODE: {} | REWARD: {} | TIME TAKEN: {} / {} \n".format(episode,np.round(episode_reward,decimals=4),final_timestep,self.max_time_steps))
					print("*"*100)
					break

			policy_grad = self.agents.get_policy_grad(episode)
			policy_grads.append(policy_grad)

		policy_grad_mat = torch.stack(policy_grads)
		mean = torch.mean(policy_grad_mat,dim=0)
		sse = torch.sum((policy_grad_mat - mean)**2,dim=-1)
		std = torch.std(sse)
		policy_grad_var = torch.var(policy_grad_mat,dim=0)

		return torch.sum(policy_grad_var).item(), std.item()
