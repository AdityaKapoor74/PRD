import os
from comet_ml import Experiment
import numpy as np
from ppo_agent_combat import PPOAgent_COMBAT
import torch
import datetime


class MAPPO_COMBAT:

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
		self.prd_num_agents = self.env.n_agents
		self.shared_num_agents = self.env._n_opponents
		self.num_actions = self.env.action_space[0].n
		self.date_time = f"{datetime.datetime.now():%d-%m-%Y}"
		self.env_name = dictionary["env"]
		self.test_num = dictionary["test_num"]
		self.max_episodes = dictionary["max_episodes"]
		self.max_time_steps = dictionary["max_time_steps"]
		self.experiment_type = dictionary["experiment_type"]
		self.update_ppo_agents = dictionary["update_ppo_agents"]
		self.prd_type = dictionary["prd_type"]


		self.comet_ml = None
		if self.save_comet_ml_plot:
			self.comet_ml = Experiment("im5zK8gFkz6j07uflhc3hXk8I",project_name=dictionary["test_num"])
			self.comet_ml.log_parameters(dictionary)


		self.agents = PPOAgent_COMBAT(self.env, dictionary, self.comet_ml)

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
			self.shared_rewards = []
			self.shared_rewards_mean_per_1000_eps = []
			self.prd_rewards = []
			self.prd_rewards_mean_per_1000_eps = []
			self.timesteps = []
			self.timesteps_mean_per_1000_eps = []

		for episode in range(1,self.max_episodes+1):

			prd_states, shared_states = self.env.reset()

			images = []

			prd_episode_reward = 0
			shared_episode_reward = 0
			final_timestep = self.max_time_steps
			for step in range(1, self.max_time_steps+1):

				if self.gif:
					# At each step, append an image to list
					if not(episode%self.gif_checkpoint):
						images.append(np.squeeze(self.env.render(mode='rgb_array')))
					# Advance a step and render a new image
					with torch.no_grad():
						prd_actions, shared_actions = self.agents.get_action(prd_states, shared_states, greedy=True)
				else:
					prd_actions, shared_actions = self.agents.get_action(prd_states, shared_states)

				prd_one_hot_actions = np.zeros((self.prd_num_agents,self.num_actions))
				for i,act in enumerate(prd_actions):
					prd_one_hot_actions[i][act] = 1

				shared_one_hot_actions = np.zeros((self.shared_num_agents,self.num_actions))
				for i,act in enumerate(shared_actions):
					shared_one_hot_actions[i][act] = 1

				prd_next_states, shared_next_states, prd_rewards, shared_rewards, prd_dones, shared_dones, info = self.env.step(prd_actions, shared_actions)

				self.agents.buffer.shared_states.append(shared_states)
				self.agents.buffer.shared_actions.append(shared_actions)
				self.agents.buffer.shared_one_hot_actions.append(shared_one_hot_actions)
				self.agents.buffer.shared_dones.append(shared_dones)
				self.agents.buffer.shared_rewards.append(shared_rewards)

				self.agents.buffer.prd_states.append(prd_states)
				self.agents.buffer.prd_actions.append(prd_actions)
				self.agents.buffer.prd_one_hot_actions.append(prd_one_hot_actions)
				self.agents.buffer.prd_dones.append(prd_dones)
				self.agents.buffer.prd_rewards.append(prd_rewards)

				shared_episode_reward += np.sum(shared_rewards)
				prd_episode_reward += np.sum(prd_rewards)

				shared_states = shared_next_states
				prd_states = prd_next_states

				if all(prd_dones):
					final_timestep = step

					print("*"*100)
					print("EPISODE: {} | TIME TAKEN: {} / {} | PRD WIN: {} | SHARED WIN: {} | DRAW: {} \n".format(episode,final_timestep,self.max_time_steps,info["agent_win"],info["opp_agent_win"],info["draw"]))
					print("SHARED REWARD: {} | NUM SHARED AGENTS ALIVE: {} | SHARED AGENTS HEALTH: {} \n".format(np.round(shared_episode_reward,decimals=4),info["num_opp_agents_alive"],info["total_opp_agents_health"]))
					print("PRD REWARD: {} | NUM PRD AGENTS ALIVE: {} | PRD AGENTS HEALTH: {} \n".format(np.round(prd_episode_reward,decimals=4),info["num_agents_alive"],info["total_agents_health"]))
					print("*"*100)

					if self.save_comet_ml_plot:
						self.comet_ml.log_metric('Episode Length', final_timestep, episode)
						self.comet_ml.log_metric('Shared Reward', shared_episode_reward, episode)
						self.comet_ml.log_metric('PRD Reward', prd_episode_reward, episode)
						self.comet_ml.log_metric('Num PRD Agents Alive', info["num_agents_alive"], episode)
						self.comet_ml.log_metric('Num Shared Agents Alive', info["num_opp_agents_alive"], episode)
						self.comet_ml.log_metric('PRD Agents Health', info["total_agents_health"], episode)
						self.comet_ml.log_metric('Shared Agents Health', info["total_opp_agents_health"], episode)
						self.comet_ml.log_metric('Shared Agent Win', info["opp_agent_win"], episode)
						self.comet_ml.log_metric('PRD Agent Win', info["agent_win"], episode)
						self.comet_ml.log_metric('Draw', info["draw"], episode)

					break

			if self.eval_policy:
				self.shared_rewards.append(shared_episode_reward)
				self.prd_rewards.append(prd_episode_reward)
				self.timesteps.append(final_timestep)

			if episode > self.save_model_checkpoint and self.eval_policy:
				self.shared_rewards_mean_per_1000_eps.append(sum(self.shared_rewards[episode-self.save_model_checkpoint:episode])/self.save_model_checkpoint)
				self.prd_rewards_mean_per_1000_eps.append(sum(self.prd_rewards[episode-self.save_model_checkpoint:episode])/self.save_model_checkpoint)
				self.timesteps_mean_per_1000_eps.append(sum(self.timesteps[episode-self.save_model_checkpoint:episode])/self.save_model_checkpoint)


			if not(episode%self.save_model_checkpoint) and episode!=0 and self.save_model:	
				torch.save(self.agents.shared_critic_network.state_dict(), self.critic_model_path+'_shared_epsiode_'+str(episode)+'.pt')
				torch.save(self.agents.shared_policy_network.state_dict(), self.actor_model_path+'_shared_epsiode_'+str(episode)+'.pt')
				torch.save(self.agents.prd_critic_network.state_dict(), self.critic_model_path+self.prd_type+'_epsiode_'+str(episode)+'.pt')
				torch.save(self.agents.prd_policy_network.state_dict(), self.actor_model_path+self.prd_type+'_epsiode_'+str(episode)+'.pt')  

			if self.learn and not(episode%self.update_ppo_agents) and episode != 0:
				self.agents.update(episode)
			if self.gif and not(episode%self.gif_checkpoint):
				print("GENERATING GIF")
				self.make_gif(np.array(images),self.gif_path)


			if self.eval_policy and episode!=0:
				np.save(os.path.join(self.policy_eval_dir,self.test_num+"_shared_reward_list"), np.array(self.shared_rewards), allow_pickle=True, fix_imports=True)
				np.save(os.path.join(self.policy_eval_dir,self.test_num+"_shared_mean_rewards_per_1000_eps"), np.array(self.shared_rewards_mean_per_1000_eps), allow_pickle=True, fix_imports=True)
				np.save(os.path.join(self.policy_eval_dir,self.test_num+"_"+self.prd_type+"_reward_list"), np.array(self.prd_rewards), allow_pickle=True, fix_imports=True)
				np.save(os.path.join(self.policy_eval_dir,self.test_num+"_"+self.prd_type+"_mean_rewards_per_1000_eps"), np.array(self.prd_rewards_mean_per_1000_eps), allow_pickle=True, fix_imports=True)
				np.save(os.path.join(self.policy_eval_dir,self.test_num+"_timestep_list"), np.array(self.timesteps), allow_pickle=True, fix_imports=True)
				np.save(os.path.join(self.policy_eval_dir,self.test_num+"_mean_timestep_per_1000_eps"), np.array(self.timesteps_mean_per_1000_eps), allow_pickle=True, fix_imports=True)

	def test(self):
		self.prd_success_rate = []
		self.shared_success_rate = []
		self.prd_rewards = []
		self.shared_rewards = []

		shared_model_path = "../../../tests/COMBAT/models/ma_gym:Combat-v0_shared_MAPPO_Q_run_1/actor_networks/actor_epsiode"
		prd_model_path = "../../../tests/COMBAT/models/ma_gym:Combat-v0_prd_above_threshold_MAPPO_Q_run_1/actor_networks/actor_epsiode"
		save_file_path = "../../../tests/COMBAT/shared_vs_prd/"

		for i in range(1000, 200000, 1000):
			self.agents.shared_policy_network.load_state_dict(torch.load(shared_model_path+str(i)+'.pt',map_location=self.device))
			self.agents.prd_policy_network.load_state_dict(torch.load(dictionary["shared_model_path_policy"],map_location=self.device))

			prd_episode_reward = 0
			shared_episode_reward = 0
			prd_succes_rate = 0
			shared_success_rate = 0

			for episode in range(1,101):
				final_timestep = self.max_time_steps
				prd_states, shared_states = self.env.reset()

				for step in range(1, self.max_time_steps+1):

					with torch.no_grad():
						prd_actions, shared_actions = self.agents.get_action(prd_states, shared_states)

					prd_next_states, shared_next_states, prd_rewards, shared_rewards, prd_dones, shared_dones, info = self.env.step(prd_actions, shared_actions)

					shared_episode_reward += np.sum(shared_rewards)
					prd_episode_reward += np.sum(prd_rewards)

					shared_states = shared_next_states
					prd_states = prd_next_states

					if all(prd_dones) or all(shared_dones):
						final_timestep = step

						print("*"*100)
						print("EPISODE: {} | TIME TAKEN: {} / {} | PRD WIN: {} | SHARED WIN: {} | DRAW: {} \n".format(episode,final_timestep,self.max_time_steps,info["agent_win"],info["opp_agent_win"],info["draw"]))
						print("SHARED REWARD: {} | NUM SHARED AGENTS ALIVE: {} | SHARED AGENTS HEALTH: {} \n".format(np.round(shared_episode_reward,decimals=4),info["num_opp_agents_alive"],info["total_opp_agents_health"]))
						print("PRD REWARD: {} | NUM PRD AGENTS ALIVE: {} | PRD AGENTS HEALTH: {} \n".format(np.round(prd_episode_reward,decimals=4),info["num_agents_alive"],info["total_agents_health"]))
						print("*"*100)

						break

			shared_episode_reward /= 100
			prd_episode_reward /= 100
			shared_success_rate /= 100
			prd_succes_rate /= 100

			self.shared_rewards.append(shared_episode_reward)
			self.prd_rewards.append(prd_episode_reward)
			self.prd_success_rate.append(prd_succes_rate)
			self.shared_success_rate.append(shared_success_rate)

		np.save(os.path.join(save_file_path+"_shared_reward_list"), np.array(self.shared_rewards), allow_pickle=True, fix_imports=True)
		np.save(os.path.join(save_file_path+"_prd_reward_list"), np.array(self.prd_rewards), allow_pickle=True, fix_imports=True)
		np.save(os.path.join(save_file_path+"_prd_success_rate_list"), np.array(self.prd_success_rate), allow_pickle=True, fix_imports=True)
		np.save(os.path.join(save_file_path+"_shared_success_rate_list"), np.array(self.shared_success_rate), allow_pickle=True, fix_imports=True)

				