import os
from comet_ml import Experiment
import numpy as np
from ppo_agent import PPOAgent
import torch
import datetime



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
		self.num_agents = self.env.n_agents
		self.num_actions = self.env.action_space[0].n
		self.date_time = f"{datetime.datetime.now():%d-%m-%Y}"
		self.env_name = dictionary["env"]
		self.test_num = dictionary["test_num"]
		self.max_episodes = dictionary["max_episodes"]
		self.max_time_steps = dictionary["max_time_steps"]
		self.experiment_type = dictionary["experiment_type"]
		self.update_ppo_agent = dictionary["update_ppo_agent"]
		self.update_type = dictionary["update_type"]
		self.model_path_value = dictionary["model_path_value"]
		self.model_path_policy = dictionary["model_path_policy"]


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

		for episode in range(1,self.max_episodes+1):

			state_agents, state_opponents = self.env.reset()

			images = []

			episode_reward = 0
			episode_goal_reached = 0
			final_timestep = self.max_time_steps
			for step in range(1, self.max_time_steps+1):

				if self.gif:
					# At each step, append an image to list
					if not(episode%self.gif_checkpoint):
						images.append(np.squeeze(self.env.render(mode='rgb_array')))
					# Advance a step and render a new image
					with torch.no_grad():
						actions = self.agents.get_action(state_agents, state_opponents, greedy=True)
					# import random
					# actions = [4 for _ in range(self.num_agents)]
					import time
					time.sleep(0.1)
				else:
					actions = self.agents.get_action(state_agents, state_opponents)

				one_hot_actions = np.zeros((self.num_agents,self.num_actions))
				for i,act in enumerate(actions):
					one_hot_actions[i][act] = 1

				next_states, rewards, dones, info = self.env.step(actions)
				next_state_agents, next_state_opponents = next_states

				episode_goal_reached += np.sum(dones)

				self.agents.buffer.state_agents.append(state_agents)
				self.agents.buffer.state_opponents.append(state_opponents)
				self.agents.buffer.actions.append(actions)
				self.agents.buffer.one_hot_actions.append(one_hot_actions)
				self.agents.buffer.dones.append(dones)
				self.agents.buffer.rewards.append(rewards)

				episode_reward += np.sum(rewards)

				state_agents = next_state_agents
				state_opponents = next_state_opponents

				if all(dones):
					final_timestep = step

					print("*"*100)
					print("EPISODE: {} | REWARD: {} | TIME TAKEN: {} / {} \n".format(episode,np.round(episode_reward,decimals=4),final_timestep,self.max_time_steps))
					print("NUM AGENTS ALIVE: {} | AGENTS HEALTH: {} | NUM OPPONENTS ALIVE: {} | OPPONENTS HEALTH: {} \n".format(info["num_agents_alive"],info["total_agents_health"],info["num_opp_agents_alive"],info["total_opp_agents_health"]))
					print("*"*100)

					if self.save_comet_ml_plot:
						self.comet_ml.log_metric('Episode_Length', final_timestep, episode)
						self.comet_ml.log_metric('Reward', episode_reward, episode)
						self.comet_ml.log_metric('Num Agents Alive', info["num_agents_alive"], episode)
						self.comet_ml.log_metric('Num Opponents Alive', info["num_opp_agents_alive"], episode)
						self.comet_ml.log_metric('Agents Health', info["total_agents_health"], episode)
						self.comet_ml.log_metric('Opponents Health', info["total_opp_agents_health"], episode)

					break

			if self.eval_policy:
				self.rewards.append(episode_reward)
				self.timesteps.append(final_timestep)

			if episode > self.save_model_checkpoint and self.eval_policy:
				self.rewards_mean_per_1000_eps.append(sum(self.rewards[episode-self.save_model_checkpoint:episode])/self.save_model_checkpoint)
				self.timesteps_mean_per_1000_eps.append(sum(self.timesteps[episode-self.save_model_checkpoint:episode])/self.save_model_checkpoint)


			if not(episode%self.save_model_checkpoint) and episode!=0 and self.save_model:	
				torch.save(self.agents.critic_network.state_dict(), self.critic_model_path+'_epsiode'+str(episode)+'.pt')
				torch.save(self.agents.policy_network.state_dict(), self.actor_model_path+'_epsiode'+str(episode)+'.pt')  

			if self.learn and not(episode%self.update_ppo_agent) and episode != 0:
				if self.update_type == "ppo":
					self.agents.update(episode) 
				elif self.update_type == "a2c":
					self.agents.a2c_update(episode)
			if self.gif and not(episode%self.gif_checkpoint):
				print("GENERATING GIF")
				self.make_gif(np.array(images),self.gif_path)


			if self.eval_policy and not(episode%self.save_model_checkpoint) and episode!=0:
				np.save(os.path.join(self.policy_eval_dir,self.test_num+"reward_list"), np.array(self.rewards), allow_pickle=True, fix_imports=True)
				np.save(os.path.join(self.policy_eval_dir,self.test_num+"mean_rewards_per_1000_eps"), np.array(self.rewards_mean_per_1000_eps), allow_pickle=True, fix_imports=True)
				np.save(os.path.join(self.policy_eval_dir,self.test_num+"timestep_list"), np.array(self.timesteps), allow_pickle=True, fix_imports=True)
				np.save(os.path.join(self.policy_eval_dir,self.test_num+"mean_timestep_per_1000_eps"), np.array(self.timesteps_mean_per_1000_eps), allow_pickle=True, fix_imports=True)
				# if "prd" in self.experiment_type:
				# 	np.save(os.path.join(self.policy_eval_dir,self.test_num+"mean_error_rate"), np.array(self.agents.error_rate), allow_pickle=True, fix_imports=True)
				# 	np.save(os.path.join(self.policy_eval_dir,self.test_num+"average_relevant_set"), np.array(self.agents.average_relevant_set), allow_pickle=True, fix_imports=True)


	def run_gradvar_exp(self, episode):
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

		for episode in range(1,self.max_episodes+1):

			state_agents, state_opponents = self.env.reset()

			
			episode_reward = 0
			episode_goal_reached = 0
			final_timestep = self.max_time_steps
			for step in range(1, self.max_time_steps+1):

				actions = self.agents.get_action(state_agents, state_opponents)

				one_hot_actions = np.zeros((self.num_agents,self.num_actions))
				for i,act in enumerate(actions):
					one_hot_actions[i][act] = 1

				next_states, rewards, dones, info = self.env.step(actions)
				next_state_agents, next_state_opponents = next_states

				episode_goal_reached += np.sum(dones)

				self.agents.buffer.state_agents.append(state_agents)
				self.agents.buffer.state_opponents.append(state_opponents)
				self.agents.buffer.actions.append(actions)
				self.agents.buffer.one_hot_actions.append(one_hot_actions)
				self.agents.buffer.dones.append(dones)
				self.agents.buffer.rewards.append(rewards)

				episode_reward += np.sum(rewards)

				state_agents = next_state_agents
				state_opponents = next_state_opponents
			
				if all(dones):
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