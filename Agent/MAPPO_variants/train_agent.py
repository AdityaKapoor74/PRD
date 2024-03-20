import os
import time
from comet_ml import Experiment
import numpy as np
from agent import PPOAgent
import torch
import datetime



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
		self.norm_returns_q = dictionary["norm_returns_q"]

		# RNN HIDDEN
		self.rnn_num_layers_q = dictionary["rnn_num_layers_q"]
		self.rnn_num_layers_v = dictionary["rnn_num_layers_v"]
		self.rnn_hidden_q = dictionary["rnn_hidden_q"]
		self.rnn_hidden_v = dictionary["rnn_hidden_v"]
		self.rnn_num_layers_actor = dictionary["rnn_num_layers_actor"]
		self.rnn_hidden_actor = dictionary["rnn_hidden_actor"]

		self.agent_ids = []
		for i in range(self.num_agents):
			agent_id = np.array([0 for i in range(self.num_agents)])
			agent_id[i] = 1
			self.agent_ids.append(agent_id)
		self.agent_ids = np.array(self.agent_ids)


		self.comet_ml = None
		if self.save_comet_ml_plot:
			self.comet_ml = Experiment("im5zK8gFkz6j07uflhc3hXk8I",project_name=dictionary["test_num"])
			self.comet_ml.log_parameters(dictionary)


		self.agents = PPOAgent(self.env, dictionary, self.comet_ml)
		# self.init_critic_hidden_state(np.zeros((1, self.num_agents, 256)))

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

		# init_rnn_hidden_v = torch.zeros(self.num_agents, self.rnn_hidden_v)
		# init_rnn_hidden_q = torch.zeros(self.num_agents, self.rnn_hidden_q)
		# init_rnn_hidden_actor = torch.zeros(self.num_agents, self.rnn_hidden_actor)

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

			images = []

			episode_reward = 0
			episode_indiv_rewards = [0 for i in range(self.num_agents)]
			final_timestep = self.max_time_steps

			if "MPE" in self.environment:
				episode_collision_rate = 0.0
				episode_goal_reached = 0.0

			
			if self.experiment_type == "prd_soft_advantage_global":
				rnn_hidden_state_q = np.zeros((self.rnn_num_layers_q, 1, self.rnn_hidden_q))
			else:
				rnn_hidden_state_q = np.zeros((self.rnn_num_layers_q, self.num_agents, self.rnn_hidden_q))
			
			
			rnn_hidden_state_v = np.zeros((self.rnn_num_layers_v, self.num_agents, self.rnn_hidden_v))
			rnn_hidden_state_actor = np.zeros((self.rnn_num_layers_actor, self.num_agents, self.rnn_hidden_actor))

			for step in range(1, self.max_time_steps+1):

				if self.gif:
					# At each step, append an image to list
					# if not(episode%self.gif_checkpoint):
					# 	images.append(np.squeeze(self.env.render(mode='rgb_array')))
					# import time
					# time.sleep(0.1)
					self.env.render()
					# Advance a step and render a new image
					with torch.no_grad():
						actions, action_logprob, next_rnn_hidden_state_actor = self.agents.get_action(states_actor, mask_actions, rnn_hidden_state_actor, greedy=False)
				else:
					actions, action_logprob, next_rnn_hidden_state_actor = self.agents.get_action(states_actor, mask_actions, rnn_hidden_state_actor)

				one_hot_actions = np.zeros((self.num_agents, self.num_actions))
				for i, act in enumerate(actions):
					one_hot_actions[i][act] = 1


				if "StarCraft" in self.environment:
					if self.experiment_type == "prd_soft_advantage_global":
						q_value, next_rnn_hidden_state_q, weights_prd, global_weights, value, next_rnn_hidden_state_v = self.agents.get_q_v_values(states_allies_critic, states_enemies_critic, one_hot_actions, rnn_hidden_state_q, rnn_hidden_state_v, indiv_dones, episode)
					else:
						q_value, next_rnn_hidden_state_q, weights_prd, value, next_rnn_hidden_state_v = self.agents.get_q_v_values(states_allies_critic, states_enemies_critic, one_hot_actions, rnn_hidden_state_q, rnn_hidden_state_v, indiv_dones, episode)
				else:
					if self.experiment_type == "prd_soft_advantage_global":
						q_value, next_rnn_hidden_state_q, weights_prd, global_weights, value, next_rnn_hidden_state_v = self.agents.get_q_v_values(states_critic, None, one_hot_actions, rnn_hidden_state_q, rnn_hidden_state_v, indiv_dones, episode)
					else:
						q_value, next_rnn_hidden_state_q, weights_prd, value, next_rnn_hidden_state_v = self.agents.get_q_v_values(states_critic, None, one_hot_actions, rnn_hidden_state_q, rnn_hidden_state_v, indiv_dones, episode)

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
					# assuming only global rewards are available
					if self.experiment_type in ["prd_soft_advantage_global", "shared", "HAPPO"]:
						rewards_to_send = [rewards]*self.num_agents
					else:
						rewards_to_send = indiv_rewards

					if self.experiment_type == "prd_soft_advantage_global":
						indiv_rewards = [rewards]*self.num_agents

						indiv_rewards = [r*c for r, c in zip(indiv_rewards, global_weights)]

						if self.norm_returns_q:
							values_shape = q_value.shape
							indiv_masks = [1-d for d in indiv_dones]
							indiv_masks = torch.FloatTensor(indiv_masks)
							q_value_denormalized = (self.agents.Q_PopArt.denormalize(torch.from_numpy(q_value).view(-1)).view(values_shape) * indiv_masks.view(values_shape)).cpu().numpy()

							q_i_value = [q*c for q, c in zip(q_value_denormalized, global_weights)]
						else:
							q_i_value = [q*c for q, c in zip(q_value, global_weights)]

						print("Agents done", indiv_dones)
						print("Global Q value", q_value[0])
						print("Q_i value", q_i_value)
						print("global weights", global_weights)
						print("-"*10)
					else:
						q_i_value = None

					if "StarCraft" in self.environment:
						self.agents.buffer.push(
							states_allies_critic, states_enemies_critic, q_value, rnn_hidden_state_q, q_i_value, weights_prd, value, rnn_hidden_state_v, \
							states_actor, rnn_hidden_state_actor, action_logprob, actions, last_one_hot_actions, one_hot_actions, mask_actions, \
							rewards_to_send, indiv_rewards, indiv_dones
							)

						states_allies_critic, states_enemies_critic = next_states_allies_critic, next_states_enemies_critic

					else:
						self.agents.buffer.push(
							states_critic, None, q_value, rnn_hidden_state_q, q_i_value, weights_prd, value, rnn_hidden_state_v, \
							states_actor, rnn_hidden_state_actor, action_logprob, actions, last_one_hot_actions, one_hot_actions, mask_actions, \
							rewards_to_send, indiv_rewards, indiv_dones
							)

						states_critic = next_states_critic

				states_actor, last_one_hot_actions, mask_actions, indiv_dones = next_states_actor, one_hot_actions, next_mask_actions, next_indiv_dones
				rnn_hidden_state_q, rnn_hidden_state_v, rnn_hidden_state_actor = next_rnn_hidden_state_q, next_rnn_hidden_state_v, next_rnn_hidden_state_actor

				if all(indiv_dones) or step == self.max_time_steps:

					final_timestep = step

					# update prd threshold
					if "threshold" in self.experiment_type:
						self.agents.update_parameters()

					if self.learn:
						# add final time to buffer
						actions, action_logprob, next_rnn_hidden_state_actor = self.agents.get_action(states_actor, mask_actions, rnn_hidden_state_actor)
					
						one_hot_actions = np.zeros((self.num_agents,self.num_actions))
						for i,act in enumerate(actions):
							one_hot_actions[i][act] = 1

						if "StarCraft" in self.environment:
							if self.experiment_type == "prd_soft_advantage_global":
								q_value, _, _, global_weights, value, _ = self.agents.get_q_v_values(states_allies_critic, states_enemies_critic, one_hot_actions, rnn_hidden_state_q, rnn_hidden_state_v, indiv_dones, episode)
							else:
								q_value, _, _, value, _ = self.agents.get_q_v_values(states_allies_critic, states_enemies_critic, one_hot_actions, rnn_hidden_state_q, rnn_hidden_state_v, indiv_dones, episode)
						else:
							if self.experiment_type == "prd_soft_advantage_global":
								q_value, _, _, global_weights, value, _ = self.agents.get_q_v_values(states_critic, None, one_hot_actions, rnn_hidden_state_q, rnn_hidden_state_v, indiv_dones, episode)
							else:
								q_value, _, _, value, _ = self.agents.get_q_v_values(states_critic, None, one_hot_actions, rnn_hidden_state_q, rnn_hidden_state_v, indiv_dones, episode)

						if self.experiment_type == "prd_soft_advantage_global":
							if self.norm_returns_q:
								values_shape = q_value.shape
								indiv_masks = [1-d for d in indiv_dones]
								indiv_masks = torch.FloatTensor(indiv_masks)
								q_value_denormalized = (self.agents.Q_PopArt.denormalize(torch.from_numpy(q_value).view(-1)).view(values_shape) * indiv_masks.view(values_shape)).cpu().numpy()

								q_i_value = [q*c for q, c in zip(q_value_denormalized, global_weights)]
							else:
								q_i_value = [q*c for q, c in zip(q_value, global_weights)]

							# if all agents die, then global_q_value and q_i_value is nan
							if all(indiv_dones):
								q_value = [0.0 for _ in range(len(q_value))]
								q_i_value = [0.0 for _ in range(len(q_i_value))]

							print("Agents done", indiv_dones)
							# print("Q value", q_value)
							print("Global Q value", q_value[0])
							print("Q_i value", q_i_value)
							# print("weight contri", approx_agent_contri_to_rew[0])
							print("global weights", global_weights)
							print("-"*10)
						else:
							q_i_value = None

						# self.agents.buffer.end_episode(final_timestep, q_value, q_i_value, value, indiv_dones)
						self.agents.buffer.end_episode(final_timestep, q_value, q_i_value, value, indiv_dones)

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
				if "prd" in self.experiment_type:
					torch.save(self.agents.critic_network_q.state_dict(), self.critic_model_path+'_Q_epsiode'+str(episode)+'.pt')
				torch.save(self.agents.critic_network_v.state_dict(), self.critic_model_path+'_V_epsiode'+str(episode)+'.pt')
				if self.experiment_type == "HAPPO":
					for i in range(self.num_agents):
						torch.save(self.agents.policy_network[i].state_dict(), self.actor_model_path+'_Policy_'+str(i)+'_epsiode'+str(episode)+'.pt')  
				else:
					torch.save(self.agents.policy_network.state_dict(), self.actor_model_path+'_epsiode'+str(episode)+'.pt')  

			if self.learn and not(episode%self.update_ppo_agent) and episode != 0:
				if self.experiment_type == "HAPPO":
					self.agents.update_HAPPO(episode)
				else:
					self.agents.update(episode)

			# elif self.gif and not(episode%self.gif_checkpoint):
			# 	print("GENERATING GIF")
			# 	self.make_gif(np.array(images),self.gif_path)


			if self.eval_policy and not(episode%self.save_model_checkpoint) and episode!=0:
				np.save(os.path.join(self.policy_eval_dir,self.test_num+"reward_list"), np.array(self.rewards), allow_pickle=True, fix_imports=True)
				np.save(os.path.join(self.policy_eval_dir,self.test_num+"mean_rewards_per_1000_eps"), np.array(self.rewards_mean_per_1000_eps), allow_pickle=True, fix_imports=True)
				np.save(os.path.join(self.policy_eval_dir,self.test_num+"timestep_list"), np.array(self.timesteps), allow_pickle=True, fix_imports=True)
				np.save(os.path.join(self.policy_eval_dir,self.test_num+"mean_timestep_per_1000_eps"), np.array(self.timesteps_mean_per_1000_eps), allow_pickle=True, fix_imports=True)
				
				if "MPE" in self.environment:
					np.save(os.path.join(self.policy_eval_dir,self.test_num+"collision_rate_list"), np.array(self.collision_rates), allow_pickle=True, fix_imports=True)
					np.save(os.path.join(self.policy_eval_dir,self.test_num+"mean_collision_rate_per_1000_eps"), np.array(self.collison_rate_mean_per_1000_eps), allow_pickle=True, fix_imports=True)


def make_env(scenario_name, benchmark=False):
	scenario = scenarios.load(scenario_name + ".py").Scenario()
	world = scenario.make_world()
	if benchmark:
		env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation, scenario.benchmark_data, scenario.isFinished)
	else:
		env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation, None, scenario.isFinished)
	return env, scenario.actor_observation_shape, scenario.critic_observation_shape


if __name__ == '__main__':

	RENDER = False
	USE_CPP_RVO2 = False

	torch.set_printoptions(profile="full")
	torch.autograd.set_detect_anomaly(True)

	for i in range(1,4):
		extension = "MAPPO_"+str(i)
		test_num = "StarCraft"
		environment = "StarCraft" # StarCraft/ MPE/ PressurePlate/ PettingZoo/ LBForaging
		if "LBForaging" in environment:
			num_players = 6
			num_food = 9
			grid_size = 12
			fully_coop = False
		env_name = "5m_vs_6m" # 5m_vs_6m/ 10m_vs_11m/ 3s5z/ crossing_team_greedy/ pressureplate-linear-6p-v0/ pursuit_v4/ "Foraging-{0}x{0}-{1}p-{2}f{3}-v2".format(grid_size, num_players, num_food, "-coop" if fully_coop else "")
		experiment_type = "HAPPO" # shared, prd_above_threshold_ascend, prd_above_threshold, prd_top_k, prd_above_threshold_decay, prd_soft_advantage prd_soft_advantage_global, HAPPO

		dictionary = {
				# TRAINING
				"iteration": i,
				"device": "gpu",
				"update_learning_rate_with_prd": False,
				"critic_dir": '../../../tests/'+test_num+'/models/'+env_name+'_'+experiment_type+'_'+extension+'/critic_networks/',
				"actor_dir": '../../../tests/'+test_num+'/models/'+env_name+'_'+experiment_type+'_'+extension+'/actor_networks/',
				"gif_dir": '../../../tests/'+test_num+'/gifs/'+env_name+'_'+experiment_type+'_'+extension+'/',
				"policy_eval_dir":'../../../tests/'+test_num+'/policy_eval/'+env_name+'_'+experiment_type+'_'+extension+'/',
				"n_epochs": 5,
				"update_ppo_agent": 30, # update ppo agent after every update_ppo_agent episodes; 10 (StarCraft/MPE/PressurePlate/LBF)/ 5 (PettingZoo)
				"environment": environment,
				"test_num": test_num,
				"extension": extension,
				"gamma": 0.99,
				"gif": False,
				"gif_checkpoint":1,
				"load_models": False,
				"model_path_value": "../../../tests/PRD_2_MPE/models/crossing_team_greedy_prd_above_threshold_MAPPO_Q_run_2/critic_networks/critic_epsiode100000.pt",
				"model_path_policy": "../../../tests/PRD_2_MPE/models/crossing_team_greedy_prd_above_threshold_MAPPO_Q_run_2/actor_networks/actor_epsiode100000.pt",
				"eval_policy": True,
				"save_model": True,
				"save_model_checkpoint": 1000,
				"save_comet_ml_plot": True,
				"learn":True,
				"warm_up": False,
				"warm_up_episodes": 500,
				"epsilon_start": 0.5,
				"epsilon_end": 0.0,
				"max_episodes": 50000, # 20000 (StarCraft environments)/ 30000 (MPE/PressurePlate)/ 5000 (PettingZoo)/ 15000 (LBForaging)
				"max_time_steps": 100, # 100 (StarCraft environments & MPE)/ 70 (PressurePlate & LBForaging)/ 500 (PettingZoo)
				"experiment_type": experiment_type,
				"parallel_training": False,
				"scheduler_need": False,
				"norm_rewards": False,
				"clamp_rewards": False,
				"clamp_rewards_value_min": 0.0,
				"clamp_rewards_value_max": 2.0,


				# ENVIRONMENT
				"env": env_name,

				# CRITIC
				"rnn_num_layers_q": 1,
				"rnn_num_layers_v": 1,
				"rnn_hidden_q": 64,
				"rnn_hidden_v": 64,				
				"q_value_lr": 5e-4, #1e-3
				"v_value_lr": 5e-4, #1e-3
				"temperature_v": 1.0,
				"temperature_q": 1.0,
				"attention_dropout_prob_q": 0.0,
				"attention_dropout_prob_v": 0.0,
				"q_weight_decay": 0.0,
				"v_weight_decay": 0.0,
				"enable_grad_clip_critic_v": True,
				"grad_clip_critic_v": 10.0,
				"enable_grad_clip_critic_q": True,
				"grad_clip_critic_q": 10.0,
				"value_clip": 0.2,
				"enable_hard_attention": False,
				"num_heads": 1,
				"critic_weight_entropy_pen": 0.0,
				"critic_weight_entropy_pen_final": 0.0,
				"critic_weight_entropy_pen_steps": 100, # number of updates
				"critic_score_regularizer": 0.0,
				"target_calc_style": "GAE", # GAE, N_steps
				"n_steps": 5,
				"norm_returns_q": True,
				"norm_returns_v": True,
				"soft_update_q": False,
				"tau_q": 0.05,
				"network_update_interval_q": 1,
				"soft_update_v": False,
				"tau_v": 0.05,
				"network_update_interval_v": 1,
				

				# ACTOR
				"use_recurrent_policy": True,
				"data_chunk_length": 10,
				"rnn_num_layers_actor": 1,
				"rnn_hidden_actor": 64,
				"enable_grad_clip_actor": True,
				"grad_clip_actor": 0.2,
				"policy_clip": 0.2,
				"policy_lr": 5e-4, #prd 1e-4
				"policy_weight_decay": 0.0,
				"entropy_pen": 1e-2, #8e-3
				"entropy_pen_final": 1e-2,
				"entropy_pen_steps": 20000,
				"gae_lambda": 0.95,
				"select_above_threshold": 0.0, 
				"threshold_min": 0.0, 
				"threshold_max": 0.2, 
				"steps_to_take": 1000,
				"top_k": 0,
				"norm_adv": True,
			}

		seeds = [42, 142, 242, 342, 442]
		torch.manual_seed(seeds[dictionary["iteration"]-1])
		if "StarCraft" in environment:
			
			import gym
			import smaclite  # noqa
			
			env = gym.make(f"smaclite/{env_name}-v0", use_cpp_rvo2=USE_CPP_RVO2)
			obs, info = env.reset(return_info=True)
			dictionary["ally_observation"] = info["ally_states"][0].shape[0]+env.n_agents #+env.action_space[0].n #4+env.action_space[0].n+env.n_agents
			dictionary["enemy_observation"] = info["enemy_states"][0].shape[0]+env.n_enemies
			dictionary["local_observation"] = obs[0].shape[0]+env.n_agents+env.action_space[0].n
			dictionary["num_agents"] = env.n_agents
			dictionary["num_actions"] = env.action_space[0].n

		elif "MPE" in environment:

			from multiagent.environment import MultiAgentEnv
			import multiagent.scenarios as scenarios

			env, actor_observation_shape, critic_observation_shape = make_env(scenario_name=dictionary["env"], benchmark=False)
			dictionary["ally_observation"] = critic_observation_shape
			dictionary["local_observation"] = actor_observation_shape+env.action_space[0].n
			dictionary["num_agents"] = env.n
			dictionary["num_actions"] = env.action_space[0].n

		elif "PressurePlate" in environment:

			import pressureplate
			import gym

			env = gym.make(env_name)
			dictionary["ally_observation"] = 133+env.n_agents
			dictionary["local_observation"] = 133+5+env.n_agents
			dictionary["num_agents"] = env.n_agents
			dictionary["num_actions"] = 5

		elif "PettingZoo" in environment:

			from pettingzoo.sisl import pursuit_v4

			num_agents = 8
			num_actions = 5
			obs_range = 7

			env = pursuit_v4.parallel_env(max_cycles=dictionary["max_time_steps"], x_size=16, y_size=16, shared_reward=False, n_evaders=30,
									n_pursuers=num_agents, obs_range=obs_range, n_catch=2, freeze_evaders=False, tag_reward=0.01,
									catch_reward=5.0, urgency_reward=-0.1, surround=True, constraint_window=1.0)
			
			dictionary["ally_observation"] = obs_range*obs_range*3 + num_agents
			dictionary["local_observation"] = obs_range*obs_range*3 + num_agents + num_actions
			dictionary["num_agents"] = num_agents
			dictionary["num_actions"] = num_actions

		elif "LBForaging" in environment:

			import lbforaging
			import gym

			env = gym.make(env_name, max_episode_steps=dictionary["max_time_steps"], penalty=0.0, normalize_reward=True)
			dictionary["ally_observation"] = num_players*3 + num_food*3 + num_players
			dictionary["local_observation"] = num_players*3 + num_food*3 + num_players + env.action_space[0].n
			dictionary["num_agents"] = num_players
			dictionary["num_actions"] = env.action_space[0].n

		ma_controller = MAPPO(env,dictionary)
		ma_controller.run()
