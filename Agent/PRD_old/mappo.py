from comet_ml import Experiment
import os
import torch
import numpy as np
from ppo_agent import PPOAgent
import datetime
import time

import gym
import smaclite  # noqa



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
		self.rnn_num_layers_actor = dictionary["rnn_num_layers_actor"]
		self.rnn_hidden_actor_dim = dictionary["rnn_hidden_actor_dim"]
		self.rnn_num_layers_critic = dictionary["rnn_num_layers_critic"]
		self.rnn_hidden_critic_dim = dictionary["rnn_hidden_critic_dim"]

		one_hot_ids = np.array([0 for i in range(self.num_agents)])
		self.agent_ids = []
		for i in range(self.num_agents):
			agent_id = one_hot_ids
			agent_id[i] = 1
			self.agent_ids.append(agent_id)
		self.agent_ids = np.array(self.agent_ids)


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

			states_actor, info = self.env.reset(return_info=True)
			mask_actions = np.array(info["avail_actions"], dtype=int)
			states_allies_critic = np.concatenate((self.agent_ids, info["ally_states"]), axis=-1)
			states_enemies_critic = info["enemy_states"]
			states_actor = np.array(states_actor)
			states_actor = np.concatenate((self.agent_ids, states_actor), axis=-1)
			last_one_hot_actions = np.zeros((self.num_agents, self.num_actions))
			indiv_dones = [0]*self.num_agents
			dones = False

			episode_reward = 0
			final_timestep = self.max_time_steps
			
			rnn_hidden_state_actor = np.zeros((self.rnn_num_layers_actor, self.num_agents, self.rnn_hidden_actor_dim))
			rnn_hidden_state_critic = np.zeros((self.rnn_num_layers_critic, self.num_agents*self.num_agents, self.rnn_hidden_critic_dim))
			
			for step in range(1, self.max_time_steps+1):

				if self.gif:
					# At each step, append an image to list
					if not(episode%self.gif_checkpoint):
						images.append(np.squeeze(self.env.render(mode='rgb_array')))
					# Advance a step and render a new image
					with torch.no_grad():
						actions, action_logprob, next_rnn_hidden_state_actor, dists = self.agents.get_action(states_actor, last_one_hot_actions, mask_actions, rnn_hidden_state_actor, greedy=True)
				else:
					actions, action_logprob, next_rnn_hidden_state_actor, dists = self.agents.get_action(states_actor, last_one_hot_actions, mask_actions, rnn_hidden_state_actor)

				one_hot_actions = np.zeros((self.num_agents,self.num_actions))
				for i,act in enumerate(actions):
					one_hot_actions[i][act] = 1

				values, next_rnn_hidden_state_critic, weights_prd = self.agents.get_critic_output(states_allies_critic, states_enemies_critic, dists, one_hot_actions, rnn_hidden_state_critic, indiv_dones)

				next_states_actor, rewards, next_dones, info = self.env.step(actions)
				indiv_rewards = info["indiv_rewards"]
				next_states_actor = np.array(next_states_actor)
				next_states_actor = np.concatenate((self.agent_ids, next_states_actor), axis=-1)
				next_states_allies_critic = np.concatenate((self.agent_ids, info["ally_states"]), axis=-1)
				next_states_enemies_critic = info["enemy_states"]
				next_mask_actions = np.array(info["avail_actions"], dtype=int)
				next_indiv_dones = np.array(info["indiv_dones"])

				if self.learn:
					self.agents.buffer.push(states_allies_critic, states_enemies_critic, values, rnn_hidden_state_critic, weights_prd, states_actor, rnn_hidden_state_actor, action_logprob, actions, dists, last_one_hot_actions, one_hot_actions, mask_actions, indiv_rewards, indiv_dones)
					
				episode_reward += np.sum(rewards)

				states_actor, last_one_hot_actions, states_allies_critic, states_enemies_critic, mask_actions, indiv_dones, dones = next_states_actor, one_hot_actions, next_states_allies_critic, next_states_enemies_critic, next_mask_actions, next_indiv_dones, next_dones
				rnn_hidden_state_actor, rnn_hidden_state_critic = next_rnn_hidden_state_actor, next_rnn_hidden_state_critic

				if dones or step == self.max_time_steps:
					
					print("*"*100)
					print("EPISODE: {} | REWARD: {} | TIME TAKEN: {} / {} \n".format(episode,np.round(episode_reward,decimals=4),step,self.max_time_steps))
					print("*"*100)

					final_timestep = step

					if self.save_comet_ml_plot:
						self.comet_ml.log_metric('Episode_Length', step, episode)
						self.comet_ml.log_metric('Reward', episode_reward, episode)
						self.comet_ml.log_metric('Num Enemies', info["num_enemies"], episode)
						self.comet_ml.log_metric('Num Allies', info["num_allies"], episode)
						self.comet_ml.log_metric('All Enemies Dead', info["all_enemies_dead"], episode)
						self.comet_ml.log_metric('All Allies Dead', info["all_allies_dead"], episode)

					if self.learn:

						if "threshold" in self.experiment_type:
							self.agents.update_parameters(episode)

						# add final time to buffer
						actions, _, _, dists = self.agents.get_action(states_actor, last_one_hot_actions, mask_actions, rnn_hidden_state_actor)
					
						one_hot_actions = np.zeros((self.num_agents,self.num_actions))
						for i,act in enumerate(actions):
							one_hot_actions[i][act] = 1

						values, _, _ = self.agents.get_critic_output(states_allies_critic, states_enemies_critic, dists, one_hot_actions, rnn_hidden_state_critic, indiv_dones)

						self.agents.buffer.end_episode(final_timestep, values, indiv_dones)

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

			if self.learn and episode % self.update_ppo_agent == 0:
				self.agents.update(episode)

			elif self.gif and not(episode%self.gif_checkpoint):
				print("GENERATING GIF")
				self.make_gif(np.array(images),self.gif_path)


		if self.eval_policy:
			np.save(os.path.join(self.policy_eval_dir,self.test_num+"reward_list"), np.array(self.rewards), allow_pickle=True, fix_imports=True)
			np.save(os.path.join(self.policy_eval_dir,self.test_num+"mean_rewards_per_1000_eps"), np.array(self.rewards_mean_per_1000_eps), allow_pickle=True, fix_imports=True)
			np.save(os.path.join(self.policy_eval_dir,self.test_num+"timestep_list"), np.array(self.timesteps), allow_pickle=True, fix_imports=True)
			np.save(os.path.join(self.policy_eval_dir,self.test_num+"mean_timestep_per_1000_eps"), np.array(self.timesteps_mean_per_1000_eps), allow_pickle=True, fix_imports=True)
