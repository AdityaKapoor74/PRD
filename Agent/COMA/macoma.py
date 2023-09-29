from comet_ml import Experiment
import os
import torch
import numpy as np
from coma_agent import COMAAgent
import datetime



class MACOMA:

	def __init__(self, env, dictionary):
		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		# self.device = "cpu"
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


		self.agents = COMAAgent(self.env, dictionary, self.comet_ml)

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

			
			self.critic_model_path = critic_dir+str(self.date_time)+'VN_ATN_FCN_lr'+str(self.agents.value_lr)+'_PN_ATN_FCN_lr'+str(self.agents.policy_lr)+'_GradNorm0.5_critic_entropy_pen'+str(self.agents.critic_entropy_pen)
			self.actor_model_path = actor_dir+str(self.date_time)+'_PN_ATN_FCN_lr'+str(self.agents.policy_lr)+'VN_SAT_FCN_lr'+str(self.agents.value_lr)+'_GradNorm0.5_critic_entropy_pen'+str(self.agents.critic_entropy_pen)
			

		if self.gif:
			gif_dir = dictionary["gif_dir"]
			try: 
				os.makedirs(gif_dir, exist_ok = True) 
				print("Gif Directory created successfully") 
			except OSError as error: 
				print("Gif Directory can not be created")
			self.gif_path = gif_dir+str(self.date_time)+'VN_SAT_FCN_lr'+str(self.agents.value_lr)+'_PN_ATN_FCN_lr'+str(self.agents.policy_lr)+'_GradNorm0.5_critic_entropy_pen'+str(self.agents.critic_entropy_pen)+'.gif'


		if self.eval_policy:
			self.policy_eval_dir = dictionary["policy_eval_dir"]
			try: 
				os.makedirs(self.policy_eval_dir, exist_ok = True) 
				print("Policy Eval Directory created successfully") 
			except OSError as error: 
				print("Policy Eval Directory can not be created")


	def update(self, trajectory, episode):

		states = torch.FloatTensor(np.array([sars[0] for sars in trajectory])).to(self.device)
		
		last_one_hot_actions = torch.FloatTensor(np.array([sars[1] for sars in trajectory])).to(self.device)
		one_hot_actions = torch.FloatTensor(np.array([sars[2] for sars in trajectory])).to(self.device)
		actions = torch.FloatTensor(np.array([sars[3] for sars in trajectory])).to(self.device)
		mask_actions = torch.FloatTensor(np.array([sars[4] for sars in trajectory])).to(self.device)
		
		rewards = torch.FloatTensor(np.array([sars[5] for sars in trajectory])).to(self.device)
		dones = torch.FloatTensor(np.array([sars[6] for sars in trajectory])).to(self.device)

		self.agents.update(states, last_one_hot_actions, one_hot_actions, actions, mask_actions, rewards, dones, episode)



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

			states, info = self.env.reset(return_info=True)
			mask_actions = (np.array(info["avail_actions"]) - 1) * 1e5
			states = np.array(states)
			states = np.concatenate((self.agent_ids, states), axis=-1)
			last_one_hot_actions = np.zeros((self.num_agents, self.num_actions))

			images = []

			trajectory = []
			episode_reward = 0
			final_timestep = self.max_time_steps
			self.agents.policy_network.rnn_hidden_state = None
			self.agents.critic_network.rnn_hidden_state = None
			self.agents.target_critic_network.rnn_hidden_state = None

			for step in range(1, self.max_time_steps+1):

				if self.gif:
					# At each step, append an image to list
					if not(episode%self.gif_checkpoint):
						images.append(np.squeeze(self.env.render(mode='rgb_array')))
					# Advance a step and render a new image
					with torch.no_grad():
						actions, _ = self.agents.get_actions(states, last_one_hot_actions, mask_actions, np.array(info["avail_actions"]))
				else:
					actions, _ = self.agents.get_actions(states, last_one_hot_actions, mask_actions, np.array(info["avail_actions"]))

				one_hot_actions = np.zeros((self.num_agents,self.num_actions))
				for i,act in enumerate(actions):
					one_hot_actions[i][act] = 1

				# rnn_hidden_state_critic = self.agents.get_critic_hidden(states, one_hot_actions)

				next_states, rewards, dones, info = self.env.step(actions)
				# dones = [int(dones)]*self.num_agents
				rewards = info["indiv_rewards"]
				next_states = np.array(next_states)
				next_states = np.concatenate((self.agent_ids, next_states), axis=-1)
				next_mask_actions = (np.array(info["avail_actions"]) - 1) * 1e5

				episode_reward += np.sum(rewards)

				# environment gives indiv stream of rewards so we make the rewards global (COMA needs global rewards)
				rewards_ = [np.sum(rewards)]*self.num_agents


				if self.learn:
					trajectory.append([states, last_one_hot_actions, one_hot_actions, actions, mask_actions, rewards_, dones])
				
				states, mask_actions, last_one_hot_actions = next_states, next_mask_actions, one_hot_actions

				if all(dones) or step == self.max_time_steps:
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
						
					
					break


			if self.eval_policy:
				self.rewards.append(episode_reward)
				self.timesteps.append(final_timestep)

			if episode > self.save_model_checkpoint and episode%self.save_model_checkpoint:
				if self.eval_policy:
					self.rewards_mean_per_1000_eps.append(sum(self.rewards[episode-self.save_model_checkpoint:episode])/self.save_model_checkpoint)
					self.timesteps_mean_per_1000_eps.append(sum(self.timesteps[episode-self.save_model_checkpoint:episode])/self.save_model_checkpoint)
					

			if not(episode%self.save_model_checkpoint) and self.save_model:	
				torch.save(self.agents.critic_network.state_dict(), self.critic_model_path+'_epsiode'+str(episode)+'.pt')
				torch.save(self.agents.policy_network.state_dict(), self.actor_model_path+'_epsiode'+str(episode)+'.pt')  

			if self.learn:
				self.update(trajectory, episode) 
			elif self.gif and not(episode%self.gif_checkpoint):
				print("GENERATING GIF")
				self.make_gif(np.array(images),self.gif_path)


		if self.eval_policy:
			np.save(os.path.join(self.policy_eval_dir,self.test_num+"reward_list"), np.array(self.rewards), allow_pickle=True, fix_imports=True)
			np.save(os.path.join(self.policy_eval_dir,self.test_num+"mean_rewards_per_1000_eps"), np.array(self.rewards_mean_per_1000_eps), allow_pickle=True, fix_imports=True)
			np.save(os.path.join(self.policy_eval_dir,self.test_num+"timestep_list"), np.array(self.timesteps), allow_pickle=True, fix_imports=True)
			np.save(os.path.join(self.policy_eval_dir,self.test_num+"mean_timestep_per_1000_eps"), np.array(self.timesteps_mean_per_1000_eps), allow_pickle=True, fix_imports=True)
