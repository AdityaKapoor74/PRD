from comet_ml import Experiment
import os
import torch
import numpy as np
from ppo_agent import PPOAgent
import datetime
import time



class MAPPO:

	def __init__(self, env, dictionary):
		if dictionary["device"] == "gpu":
			self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		else:
			self.device = "cpu"
		
		self.env = env
		self.environment = dictionary["environment"]
		self.gif = dictionary["gif"]
		self.save_model = dictionary["save_model"]
		self.save_model_checkpoint = dictionary["save_model_checkpoint"]
		self.save_comet_ml_plot = dictionary["save_comet_ml_plot"]
		self.learn = dictionary["learn"]
		self.gif_checkpoint = dictionary["gif_checkpoint"]
		self.eval_policy = dictionary["eval_policy"]
		self.num_agents = dictionary["num_agents"]
		
		if "StarCraft" in self.environment:
			self.num_enemies = self.env.n_enemies

			self.enemy_ids = []
			for i in range(self.num_enemies):
				enemy_id = np.array([0 for i in range(self.num_enemies)])
				enemy_id[i] = 1
				self.enemy_ids.append(enemy_id)
			self.enemy_ids = np.array(self.enemy_ids)
		else:
			self.num_enemies = 1

		self.num_actions = dictionary["num_actions"]
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


	# FOR COLLISION AVOIDANCE ENVIRONMENT
	def split_states(self, states):
		states_critic = []
		states_actor = []
		for i in range(self.num_agents):
			states_critic.append(states[i][0])
			states_actor.append(states[i][1])

		states_critic = np.asarray(states_critic)
		states_actor = np.asarray(states_actor)

		return states_critic, states_actor


	def preprocess_state(self, states):
		states = np.array(states)
		# bring agent states first and then food locations
		states_ = np.concatenate([states[:, -self.num_agents*3:], states[:, :-self.num_agents*3]], axis=-1)
		for curr_agent_num in range(states_.shape[0]):
			curr_px, curr_py = states_[curr_agent_num][0], states_[curr_agent_num][1]
			# looping over the state
			for i in range(3, states_[curr_agent_num].shape[0], 3):
				states_[curr_agent_num][i], states_[curr_agent_num][i+1] = states_[curr_agent_num][i]-curr_px, states_[curr_agent_num][i+1]-curr_py
		return states_


	def run(self):  
		if self.eval_policy:
			self.rewards = []
			self.rewards_mean_per_1000_eps = []
			self.timesteps = []
			self.timesteps_mean_per_1000_eps = []

			if "MPE" in self.environment:
				self.collision_rates = []
				self.collison_rate_mean_per_1000_eps = []
		

		for episode in range(1,self.max_episodes+1):

			if "StarCraft" in self.environment:
				states_actor, info = self.env.reset(return_info=True)
				mask_actions = np.array(info["avail_actions"], dtype=int)
				last_one_hot_actions = np.zeros((self.num_agents, self.num_actions))
				states_allies_critic = np.concatenate((self.agent_ids, info["ally_states"]), axis=-1)
				states_enemies_critic = np.concatenate((self.enemy_ids, info["enemy_states"]), axis=-1)
				states_actor = np.array(states_actor)
				states_actor = np.concatenate((self.agent_ids, states_actor, last_one_hot_actions), axis=-1)
				indiv_dones = [0]*self.num_agents
				indiv_dones = np.array(indiv_dones)
			elif "MPE" in self.environment:
				states = self.env.reset()
				states_critic, states_actor = self.split_states(states)
				last_one_hot_actions = np.zeros((self.num_agents, self.num_actions))
				states_actor = np.concatenate((states_actor, last_one_hot_actions), axis=-1)
				indiv_dones = [0]*self.num_agents
				indiv_dones = np.array(indiv_dones)
				mask_actions = np.ones((self.num_agents, self.num_actions))
			elif "PressurePlate" in self.environment:
				states = self.env.reset()
				last_one_hot_actions = np.zeros((self.num_agents, self.num_actions))
				states_critic = np.array(states)
				states_critic = np.concatenate((self.agent_ids, states_critic), axis=-1)
				states_actor = np.array(states)
				states_actor = np.concatenate((self.agent_ids, states_actor, last_one_hot_actions), axis=-1)
				indiv_dones = [0]*self.num_agents
				indiv_dones = np.array(indiv_dones)
				mask_actions = np.ones((self.num_agents, self.num_actions))
			elif "PettingZoo" in self.environment:
				pz_state, info = self.env.reset()
				states = np.array([s for s in pz_state.values()]).reshape(self.num_agents, -1)
				last_one_hot_actions = np.zeros((self.num_agents, self.num_actions))
				states_critic = np.concatenate((self.agent_ids, states), axis=-1)
				states_actor = np.concatenate((self.agent_ids, states, last_one_hot_actions), axis=-1)
				indiv_dones = [0]*self.num_agents
				indiv_dones = np.array(indiv_dones)
				mask_actions = np.ones((self.num_agents, self.num_actions))
			elif "LBForaging" in self.environment:
				states = self.preprocess_state(self.env.reset())
				last_one_hot_actions = np.zeros((self.num_agents, self.num_actions))
				states_critic = np.concatenate((self.agent_ids, states), axis=-1)
				states_actor = np.concatenate((self.agent_ids, states, last_one_hot_actions), axis=-1)
				indiv_dones = [0]*self.num_agents
				indiv_dones = np.array(indiv_dones)
				mask_actions = np.ones((self.num_agents, self.num_actions))

			episode_reward = 0
			episode_indiv_rewards = [0 for i in range(self.num_agents)]
			final_timestep = self.max_time_steps

			if "MPE" in self.environment:
				episode_collision_rate = 0.0
				episode_goal_reached = 0.0
			
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

				if "StarCraft" in self.environment:
					values, next_rnn_hidden_state_critic, weights_prd = self.agents.get_critic_output(states_allies_critic, states_enemies_critic, dists, one_hot_actions, rnn_hidden_state_critic, indiv_dones)
				else:
					values, next_rnn_hidden_state_critic, weights_prd = self.agents.get_critic_output(states_critic, None, dists, one_hot_actions, rnn_hidden_state_critic, indiv_dones)

				if "StarCraft" in self.environment:
					next_states_actor, rewards, next_dones, info = self.env.step(actions)
					next_states_actor = np.array(next_states_actor)
					next_states_actor = np.concatenate((self.agent_ids, next_states_actor, one_hot_actions), axis=-1)
					next_states_allies_critic = np.concatenate((self.agent_ids, info["ally_states"]), axis=-1)
					next_states_enemies_critic = np.concatenate((self.enemy_ids, info["enemy_states"]), axis=-1)
					next_mask_actions = np.array(info["avail_actions"], dtype=int)
					next_indiv_dones = info["indiv_dones"]
					indiv_rewards = info["indiv_rewards"]

				elif "MPE" in self.environment:
					next_states, rewards, next_indiv_dones, info = self.env.step(actions)
					next_states_critic, next_states_actor = self.split_states(next_states)
					next_states_actor = np.concatenate((next_states_actor, one_hot_actions), axis=-1)
					next_mask_actions = np.ones((self.num_agents, self.num_actions))
					collision_rate = [value[1] for value in rewards]
					goal_reached = [value[2] for value in rewards]
					indiv_rewards = [value[0] for value in rewards] 
					episode_collision_rate += np.sum(collision_rate)
					episode_goal_reached += np.sum(goal_reached)
					next_dones = all(next_indiv_dones)
					rewards = np.sum(indiv_rewards)

				elif "PressurePlate" in self.environment:
					next_states, indiv_rewards, next_indiv_dones, info = self.env.step(actions)
					next_states_actor = np.array(next_states)
					next_states_actor = np.concatenate((self.agent_ids, next_states_actor, one_hot_actions), axis=-1)
					next_states_critic = np.array(next_states)
					next_states_critic = np.concatenate((self.agent_ids, next_states_critic), axis=-1)
					rewards = np.sum(indiv_rewards)
					next_mask_actions = np.ones((self.num_agents, self.num_actions))
					next_dones = all(next_indiv_dones)

				elif "PettingZoo" in self.environment:
					actions_dict = {}
					for i, agent in enumerate(self.env.agents):
						actions_dict[agent] = actions[i]
					pz_next_states, pz_rewards, pz_dones, truncation, info = self.env.step(actions_dict)
					next_states = np.array([s for s in pz_next_states.values()]).reshape(self.num_agents, -1)
					indiv_rewards = np.array([s for s in pz_rewards.values()])
					next_indiv_dones = np.array([s for s in pz_dones.values()])
					next_states_actor = np.concatenate((self.agent_ids, next_states, one_hot_actions), axis=-1)
					next_states_critic = np.concatenate((self.agent_ids, next_states), axis=-1)
					rewards = np.sum(indiv_rewards)
					next_mask_actions = np.ones((self.num_agents, self.num_actions))
					next_dones = all(next_indiv_dones)

				elif "LBForaging" in self.environment:
					next_states, indiv_rewards, next_indiv_dones, info = self.env.step(actions)
					next_states = self.preprocess_state(next_states)
					next_states_actor = np.concatenate((self.agent_ids, next_states, one_hot_actions), axis=-1)
					next_states_critic = np.concatenate((self.agent_ids, next_states), axis=-1)
					num_food_left = info["num_food_left"]
					rewards = np.sum(indiv_rewards)
					next_mask_actions = np.ones((self.num_agents, self.num_actions))
					next_dones = all(next_indiv_dones)

				episode_reward += np.sum(rewards)
				episode_indiv_rewards = [r+indiv_rewards[i] for i, r in enumerate(episode_indiv_rewards)]

				if self.learn:
					if "StarCraft" in self.environment:
						self.agents.buffer.push(
							states_allies_critic, states_enemies_critic, values, rnn_hidden_state_critic, weights_prd, \
							states_actor, rnn_hidden_state_actor, action_logprob, actions, dists, last_one_hot_actions, one_hot_actions, mask_actions, \
							indiv_rewards, indiv_dones
							)

						states_allies_critic, states_enemies_critic = next_states_allies_critic, next_states_enemies_critic

					else:
						self.agents.buffer.push(
							states_critic, None, values, rnn_hidden_state_critic, weights_prd, \
							states_actor, rnn_hidden_state_actor, action_logprob, actions, dists, last_one_hot_actions, one_hot_actions, mask_actions, \
							indiv_rewards, indiv_dones
							)

						states_critic = next_states_critic

				states_actor, last_one_hot_actions, mask_actions, indiv_dones, dones = next_states_actor, one_hot_actions, next_mask_actions, next_indiv_dones, next_dones
				rnn_hidden_state_critic, rnn_hidden_state_actor = next_rnn_hidden_state_critic, next_rnn_hidden_state_actor

				

				if dones or step == self.max_time_steps:

					if self.learn:

						if "threshold" in self.experiment_type:
							self.agents.update_parameters(episode)

						# add final time to buffer
						actions, _, _, dists = self.agents.get_action(states_actor, last_one_hot_actions, mask_actions, rnn_hidden_state_actor)
					
						one_hot_actions = np.zeros((self.num_agents,self.num_actions))
						for i,act in enumerate(actions):
							one_hot_actions[i][act] = 1

						if "StarCraft" in self.environment:
							values, _, _ = self.agents.get_critic_output(states_allies_critic, states_enemies_critic, dists, one_hot_actions, rnn_hidden_state_critic, indiv_dones)
						else:
							values, _, _ = self.agents.get_critic_output(states_critic, None, dists, one_hot_actions, rnn_hidden_state_critic, indiv_dones)

						self.agents.buffer.end_episode(final_timestep, values, indiv_dones)

					
					print("*"*100)
					print("EPISODE: {} | REWARD: {} | TIME TAKEN: {} / {} | INDIV REWARD STREAMS: {} \n".format(episode, np.round(episode_reward,decimals=4), step, self.max_time_steps, episode_indiv_rewards))
					
					if "StarCraft" in self.environment:
						print("Num Allies Alive: {} | Num Enemies Alive: {} | AGENTS DEAD: {} \n".format(info["num_allies"], info["num_enemies"], info["indiv_dones"]))
					elif "MPE" in self.environment:
						print("Num Agents Reached Goal {} | Num Collisions {} \n".format(np.sum(indiv_dones), episode_collision_rate))
					elif "PressurePlate" in self.environment:
						print("Num Agents Reached Goal {} \n".format(np.sum(indiv_dones)))
					print("*"*100)

					if self.save_comet_ml_plot:
						self.comet_ml.log_metric('Episode_Length', step, episode)
						self.comet_ml.log_metric('Reward', episode_reward, episode)

						if "StarCraft" in self.environment:
							self.comet_ml.log_metric('Num Enemies', info["num_enemies"], episode)
							self.comet_ml.log_metric('Num Allies', info["num_allies"], episode)
							self.comet_ml.log_metric('All Enemies Dead', info["all_enemies_dead"], episode)
							self.comet_ml.log_metric('All Allies Dead', info["all_allies_dead"], episode)
						elif "MPE" in self.environment:
							self.comet_ml.log_metric('Number of Collision', episode_collision_rate, episode)
							self.comet_ml.log_metric('Num Agents Goal Reached', np.sum(indiv_dones), episode)
						elif "PressurePlate" in self.environment:
							self.comet_ml.log_metric('Num Agents Goal Reached', np.sum(indiv_dones), episode)
						elif "LBForaging" in self.environment:
							self.comet_ml.log_metric('Num Food Left', num_food_left, episode)

					
					break
					

			if self.agents.scheduler_need:
				if self.experiment_type == "HAPPO":
					for i in range(self.num_agents):
						self.agents.scheduler_policy[i].step()
				else:
					self.agents.scheduler_policy.step()
				self.agents.scheduler_q_critic.step()
				self.agents.scheduler_v_critic.step()

			if self.eval_policy:
				self.rewards.append(episode_reward)
				self.timesteps.append(final_timestep)

				if "MPE" in self.environment:
					self.collision_rates.append(episode_collision_rate)

			if episode > self.save_model_checkpoint and self.eval_policy:
				self.rewards_mean_per_1000_eps.append(sum(self.rewards[episode-self.save_model_checkpoint:episode])/self.save_model_checkpoint)
				self.timesteps_mean_per_1000_eps.append(sum(self.timesteps[episode-self.save_model_checkpoint:episode])/self.save_model_checkpoint)

				if "MPE" in self.environment:
					self.collison_rate_mean_per_1000_eps.append(sum(self.collision_rates[episode-self.save_model_checkpoint:episode])/self.save_model_checkpoint)
				

			if not(episode%self.save_model_checkpoint) and episode!=0 and self.save_model:	
				torch.save(self.agents.critic_network.state_dict(), self.critic_model_path+'_epsiode'+str(episode)+'.pt')
				torch.save(self.agents.policy_network.state_dict(), self.actor_model_path+'_epsiode'+str(episode)+'.pt')  

			if self.learn and episode % self.update_ppo_agent == 0:
				self.agents.update(episode)

			elif self.gif and not(episode%self.gif_checkpoint):
				print("GENERATING GIF")
				self.make_gif(np.array(images),self.gif_path)


		if self.eval_policy and not(episode%self.save_model_checkpoint) and episode!=0:
				np.save(os.path.join(self.policy_eval_dir,self.test_num+"reward_list"), np.array(self.rewards), allow_pickle=True, fix_imports=True)
				np.save(os.path.join(self.policy_eval_dir,self.test_num+"mean_rewards_per_1000_eps"), np.array(self.rewards_mean_per_1000_eps), allow_pickle=True, fix_imports=True)
				np.save(os.path.join(self.policy_eval_dir,self.test_num+"timestep_list"), np.array(self.timesteps), allow_pickle=True, fix_imports=True)
				np.save(os.path.join(self.policy_eval_dir,self.test_num+"mean_timestep_per_1000_eps"), np.array(self.timesteps_mean_per_1000_eps), allow_pickle=True, fix_imports=True)
				
				if "MPE" in self.environment:
					np.save(os.path.join(self.policy_eval_dir,self.test_num+"collision_rate_list"), np.array(self.collision_rates), allow_pickle=True, fix_imports=True)
					np.save(os.path.join(self.policy_eval_dir,self.test_num+"mean_collision_rate_per_1000_eps"), np.array(self.collison_rate_mean_per_1000_eps), allow_pickle=True, fix_imports=True)
