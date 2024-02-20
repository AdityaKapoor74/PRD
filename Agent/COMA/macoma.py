from comet_ml import Experiment
import os
import torch
import numpy as np
from coma_agent import COMAAgent
import datetime
from buffer import RolloutBuffer



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
		self.update_episode_interval = dictionary["update_episode_interval"]

		self.actor_rnn_num_layers = dictionary["actor_rnn_num_layers"]
		self.actor_rnn_hidden_dim = dictionary["actor_rnn_hidden_dim"]
		self.critic_rnn_num_layers = dictionary["critic_rnn_num_layers"]
		self.critic_rnn_hidden_dim = dictionary["critic_rnn_hidden_dim"]

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

		self.buffer = RolloutBuffer(
			num_episodes = self.update_episode_interval,
			max_time_steps = self.max_time_steps,
			num_agents = self.num_agents,
			critic_obs_shape = dictionary["global_observation"],
			actor_obs_shape = dictionary["local_observation"],
			critic_rnn_num_layers=dictionary["critic_rnn_num_layers"],
			actor_rnn_num_layers=dictionary["actor_rnn_num_layers"],
			critic_rnn_hidden_state_dim=dictionary["critic_rnn_hidden_dim"],
			actor_rnn_hidden_state_dim=dictionary["actor_rnn_hidden_dim"],
			data_chunk_length=dictionary["data_chunk_length"],
			num_actions = self.num_actions,
			lambda_=dictionary["lambda"],
			gamma=dictionary["gamma"],
			)

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
			action_masks = np.array(info["avail_actions"])
			states = np.array(states)
			states = np.concatenate((self.agent_ids, states), axis=-1)
			last_one_hot_actions = np.zeros((self.num_agents, self.num_actions))
			states_allies = np.concatenate((self.agent_ids, info["ally_states"]), axis=-1)
			states_enemies = np.repeat(np.array(info["enemy_states"]).reshape(1, -1), self.num_agents, axis=0)
			full_state = np.concatenate((states_allies, states_enemies), axis=-1)
			dones = [0]*self.num_agents
			indiv_dones = [0]*self.num_agents
			indiv_dones = np.array(indiv_dones)

			images = []

			trajectory = []
			episode_reward = 0
			final_timestep = self.max_time_steps
			# self.agents.policy_network.rnn_hidden_state = None
			# self.agents.critic_network.rnn_hidden_state = None
			# self.agents.target_critic_network.rnn_hidden_state = None

			actor_rnn_hidden_state = np.zeros((self.actor_rnn_num_layers, self.num_agents, self.actor_rnn_hidden_dim))
			critic_rnn_hidden_state = np.zeros((self.actor_rnn_num_layers, self.num_agents, self.critic_rnn_hidden_dim))

			for step in range(1, self.max_time_steps+1):

				if self.gif:
					# At each step, append an image to list
					if not(episode%self.gif_checkpoint):
						images.append(np.squeeze(self.env.render(mode='rgb_array')))
					# Advance a step and render a new image
					with torch.no_grad():
						probs, actions, next_actor_rnn_hidden_state = self.agents.get_actions(states, last_one_hot_actions, actor_rnn_hidden_state, action_masks)
				else:
					probs, actions, next_actor_rnn_hidden_state = self.agents.get_actions(states, last_one_hot_actions, actor_rnn_hidden_state, action_masks)

				next_last_one_hot_actions = np.zeros((self.num_agents,self.num_actions))
				for i,act in enumerate(actions):
					next_last_one_hot_actions[i][act] = 1

				q_value, next_critic_rnn_hidden_state = self.agents.get_critic_output(full_state, next_last_one_hot_actions, critic_rnn_hidden_state)

				next_states, rewards, dones, info = self.env.step(actions)
				next_dones = [int(dones)]*self.num_agents
				# rewards = info["indiv_rewards"]
				next_states = np.array(next_states)
				next_states = np.concatenate((self.agent_ids, next_states), axis=-1)
				next_action_masks = np.array(info["avail_actions"])
				next_states_allies = np.concatenate((self.agent_ids, info["ally_states"]), axis=-1)
				next_states_enemies = np.repeat(np.array(info["enemy_states"]).reshape(1, -1), self.num_agents, axis=0)
				next_full_state = np.concatenate((next_states_allies, next_states_enemies), axis=-1)
				next_indiv_dones = info["indiv_dones"]

				episode_reward += np.sum(rewards)

				# environment gives indiv stream of rewards so we make the rewards global (COMA needs global rewards)
				rewards_ = [np.sum(rewards)]*self.num_agents


				if self.learn:
					self.buffer.push(full_state, states, critic_rnn_hidden_state, actor_rnn_hidden_state, q_value, last_one_hot_actions, probs, actions, next_last_one_hot_actions, action_masks, rewards, dones)
				
				states, full_state, action_masks, last_one_hot_actions, dones, indiv_dones = next_states, next_full_state, next_action_masks, next_last_one_hot_actions, next_dones, next_indiv_dones

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

			if self.learn and episode !=0 and episode%self.update_episode_interval == 0:
				self.agents.update(self.buffer, episode)
				self.buffer.clear()
			elif self.gif and not(episode%self.gif_checkpoint):
				print("GENERATING GIF")
				self.make_gif(np.array(images),self.gif_path)


		if self.eval_policy:
			np.save(os.path.join(self.policy_eval_dir,self.test_num+"reward_list"), np.array(self.rewards), allow_pickle=True, fix_imports=True)
			np.save(os.path.join(self.policy_eval_dir,self.test_num+"mean_rewards_per_1000_eps"), np.array(self.rewards_mean_per_1000_eps), allow_pickle=True, fix_imports=True)
			np.save(os.path.join(self.policy_eval_dir,self.test_num+"timestep_list"), np.array(self.timesteps), allow_pickle=True, fix_imports=True)
			np.save(os.path.join(self.policy_eval_dir,self.test_num+"mean_timestep_per_1000_eps"), np.array(self.timesteps_mean_per_1000_eps), allow_pickle=True, fix_imports=True)
