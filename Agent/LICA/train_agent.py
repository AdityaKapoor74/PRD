import os
import time
from comet_ml import Experiment
import numpy as np
from agent import LICAAgent
from buffer import RolloutBuffer
import torch
import datetime



class LICA:

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
		self.num_actions = dictionary["num_actions"]
		self.date_time = f"{datetime.datetime.now():%d-%m-%Y}"
		self.env_name = dictionary["env"]
		self.test_num = dictionary["test_num"]
		self.max_episodes = dictionary["max_episodes"]
		self.max_time_steps = dictionary["max_time_steps"]
		self.update_episode_interval = dictionary["update_episode_interval"]

		self.rnn_num_layers = dictionary["rnn_num_layers"]
		self.rnn_hidden_dim = dictionary["rnn_hidden_dim"]
		self.data_chunk_length = dictionary["data_chunk_length"]
		self.critic_obs_shape = dictionary["global_observation"]
		self.actor_obs_shape = dictionary["local_observation"]

		one_hot_ids = np.array([0 for i in range(self.num_agents)])
		self.agent_ids = []
		for i in range(self.num_agents):
			agent_id = one_hot_ids
			agent_id[i] = 1
			self.agent_ids.append(agent_id)
		self.agent_ids = np.array(self.agent_ids)

		self.buffer = RolloutBuffer(
			num_episodes = self.update_episode_interval,
			max_time_steps = self.max_time_steps,
			num_agents = self.num_agents,
			critic_obs_shape = dictionary["global_observation"],
			actor_obs_shape = dictionary["local_observation"],
			num_actions = self.num_actions,
			rnn_num_layers = self.rnn_num_layers,
			rnn_hidden_dim = self.rnn_hidden_dim,
			data_chunk_length = self.data_chunk_length,
			gamma=dictionary["gamma"],
			lambda_=dictionary["lambda"],
			device=self.device,
			)

		self.comet_ml = None
		if self.save_comet_ml_plot:
			self.comet_ml = Experiment("im5zK8gFkz6j07uflhc3hXk8I",project_name=dictionary["test_num"])
			self.comet_ml.log_parameters(dictionary)


		self.agents = LICAAgent(self.env, dictionary, self.comet_ml)

		if self.save_model:
			model_dir = dictionary["model_dir"]
			try: 
				os.makedirs(model_dir, exist_ok = True) 
				print("Model Directory created successfully") 
			except OSError as error: 
				print("Model Directory cannot be created")
			
			self.model_path = model_dir+"model"
			

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

	# FOR LBF
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

		for episode in range(1, self.max_episodes+1):

			# states, info = self.env.reset(return_info=True)
			# mask_actions = (np.array(info["avail_actions"]) - 1) * 1e5
			# states = np.array(states)
			# states = np.concatenate((self.agent_ids, states), axis=-1)
			# last_one_hot_action = np.zeros((self.num_agents, self.num_actions))
			# states_allies = np.concatenate((self.agent_ids, info["ally_states"]), axis=-1).reshape(-1)
			# states_enemies = np.array(info["enemy_states"]).reshape(-1)
			# full_state = np.concatenate((states_allies, states_enemies), axis=-1)
			# indiv_dones = [0]*self.num_agents
			# indiv_dones = np.array(indiv_dones)

			if "StarCraft" in self.environment:
				states, info = self.env.reset(return_info=True)
				mask_actions = np.array(info["avail_actions"]) #(np.array(info["avail_actions"]) - 1) * 1e5
				last_one_hot_action = np.zeros((self.num_agents, self.num_actions))
				states = np.array(states)
				states = np.concatenate((self.agent_ids, states), axis=-1)
				states_allies = np.concatenate((self.agent_ids, info["ally_states"]), axis=-1).reshape(-1)
				states_enemies = np.array(info["enemy_states"]).reshape(-1)
				full_state = np.concatenate((states_allies, states_enemies), axis=-1)
				indiv_dones = [0]*self.num_agents
				indiv_dones = np.array(indiv_dones)
				dones = all(indiv_dones)
			elif "MPE" in self.environment:
				states = self.env.reset()
				full_state, states = self.split_states(states)
				full_state = full_state.reshape(-1)
				last_one_hot_action = np.zeros((self.num_agents, self.num_actions))
				states = np.concatenate((states, last_one_hot_action), axis=-1)
				indiv_dones = [0]*self.num_agents
				indiv_dones = np.array(indiv_dones)
				dones = all(indiv_dones)
				mask_actions = np.ones((self.num_agents, self.num_actions))
			elif "PressurePlate" in self.environment:
				states = self.env.reset()
				last_one_hot_action = np.zeros((self.num_agents, self.num_actions))
				full_state = np.array(states)
				full_state = np.concatenate((self.agent_ids, states), axis=-1).reshape(-1)
				states = np.array(states)
				states = np.concatenate((self.agent_ids, states, last_one_hot_action), axis=-1)
				indiv_dones = [0]*self.num_agents
				indiv_dones = np.array(indiv_dones)
				dones = all(indiv_dones)
				mask_actions = np.ones((self.num_agents, self.num_actions))
			elif "PettingZoo" in self.environment:
				pz_state, info = self.env.reset()
				states = np.array([s for s in pz_state.values()]).reshape(self.num_agents, -1)
				last_one_hot_action = np.zeros((self.num_agents, self.num_actions))
				full_state = np.concatenate((self.agent_ids, states), axis=-1).reshape(-1)
				states = np.concatenate((self.agent_ids, states, last_one_hot_action), axis=-1)
				indiv_dones = [0]*self.num_agents
				indiv_dones = np.array(indiv_dones)
				dones = all(indiv_dones)
				mask_actions = np.ones((self.num_agents, self.num_actions))
			elif "LBForaging" in self.environment:
				states = self.preprocess_state(self.env.reset())
				last_one_hot_action = np.zeros((self.num_agents, self.num_actions))
				full_state = np.concatenate((self.agent_ids, states), axis=-1).reshape(-1)
				states = np.concatenate((self.agent_ids, states, last_one_hot_action), axis=-1)
				indiv_dones = [0]*self.num_agents
				indiv_dones = np.array(indiv_dones)
				dones = all(indiv_dones)
				mask_actions = np.ones((self.num_agents, self.num_actions))

			images = []

			episode_reward = 0
			episode_indiv_rewards = [0 for i in range(self.num_agents)]
			final_timestep = self.max_time_steps

			if "MPE" in self.environment:
				episode_collision_rate = 0.0
				episode_goal_reached = 0.0

			# self.agents.actor.rnn_hidden_obs = None
			rnn_hidden_state = np.zeros((self.rnn_num_layers, self.num_agents, self.rnn_hidden_dim))

			for step in range(1, self.max_time_steps+1):

				if self.gif:
					# At each step, append an image to list
					if not(episode%self.gif_checkpoint):
						images.append(np.squeeze(self.env.render(mode='rgb_array')))
					import time
					# Advance a step and render a new image
					with torch.no_grad():
						actions, next_rnn_hidden_state = self.agents.get_action(states, rnn_hidden_state, mask_actions)
				else:
					actions, next_rnn_hidden_state = self.agents.get_action(states, rnn_hidden_state, mask_actions)

				next_last_one_hot_action = np.zeros((self.num_agents, self.num_actions))
				for i,act in enumerate(actions):
					next_last_one_hot_action[i][act] = 1

				# next_states, rewards, dones, info = self.env.step(actions)
				# # dones = [int(dones)]*self.num_agents
				# # rewards = info["indiv_rewards"]
				# next_states = np.array(next_states)
				# next_states = np.concatenate((self.agent_ids, next_states), axis=-1)
				# next_mask_actions = (np.array(info["avail_actions"]) - 1) * 1e5
				# next_states_allies = np.concatenate((self.agent_ids, info["ally_states"]), axis=-1).reshape(-1)
				# next_states_enemies = np.array(info["enemy_states"]).reshape(-1)
				# next_full_state = np.concatenate((next_states_allies, next_states_enemies), axis=-1)

				if "StarCraft" in self.environment:
					next_states, rewards, next_dones, info = self.env.step(actions)
					next_states = np.array(next_states)
					next_states = np.concatenate((self.agent_ids, next_states), axis=-1)
					next_states_allies = np.concatenate((self.agent_ids, info["ally_states"]), axis=-1).reshape(-1)
					next_states_enemies = np.array(info["enemy_states"]).reshape(-1)
					next_full_state = np.concatenate((next_states_allies, next_states_enemies), axis=-1)
					next_mask_actions = np.array(info["avail_actions"]) # (np.array(info["avail_actions"]) - 1) * 1e5
					next_indiv_dones = info["indiv_dones"]

				elif "MPE" in self.environment:
					next_states, rewards, next_indiv_dones, info = self.env.step(actions)
					next_full_state, next_states = self.split_states(next_states)
					next_full_state = next_full_state.reshape(-1)
					next_states = np.concatenate((next_states, next_last_one_hot_action), axis=-1)
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
					next_full_state = np.array(next_states)
					next_full_state = np.concatenate((self.agent_ids, next_full_state), axis=-1).reshape(-1)
					next_states = np.array(next_states)
					next_states = np.concatenate((self.agent_ids, next_states, next_last_one_hot_action), axis=-1)
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
					next_full_state = np.concatenate((self.agent_ids, next_states), axis=-1).reshape(-1)
					next_states = np.concatenate((self.agent_ids, next_states, next_last_one_hot_action), axis=-1)
					rewards = np.sum(indiv_rewards)
					next_mask_actions = np.ones((self.num_agents, self.num_actions))
					next_dones = all(next_indiv_dones)

				elif "LBForaging" in self.environment:
					next_states, indiv_rewards, next_indiv_dones, info = self.env.step(actions)
					next_states = self.preprocess_state(next_states)
					next_full_state = np.concatenate((self.agent_ids, next_states), axis=-1).reshape(-1)
					next_states = np.concatenate((self.agent_ids, next_states, next_last_one_hot_action), axis=-1)
					num_food_left = info["num_food_left"]
					rewards = np.sum(indiv_rewards)
					next_mask_actions = np.ones((self.num_agents, self.num_actions))
					next_dones = all(next_indiv_dones)

				if not self.gif:
					self.buffer.push(full_state, states, rnn_hidden_state, actions, next_last_one_hot_action, mask_actions, rewards, dones)

				states, full_state, mask_actions, last_one_hot_action, rnn_hidden_state, dones, indiv_dones = next_states, next_full_state, next_mask_actions, next_last_one_hot_action, next_rnn_hidden_state, next_dones, next_indiv_dones

				episode_reward += np.sum(rewards)

				if dones or step == self.max_time_steps:

					print("*"*100)
					print("EPISODE: {} | REWARD: {} | TIME TAKEN: {} / {} \n".format(episode, np.round(episode_reward,decimals=4), step, self.max_time_steps))
					
					if "StarCraft" in self.environment:
						print("Num Allies Alive: {} | Num Enemies Alive: {} | AGENTS DEAD: {} \n".format(info["num_allies"], info["num_enemies"], info["indiv_dones"]))
					elif "MPE" in self.environment:
						print("Num Agents Reached Goal {} | Num Collisions {} \n".format(np.sum(indiv_dones), episode_collision_rate))
					elif "PressurePlate" in self.environment:
						print("Num Agents Reached Goal {} \n".format(np.sum(indiv_dones)))
					print("*"*100)

					final_timestep = step

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

					
					# add final time to buffer
					actions, _ = self.agents.get_action(states, rnn_hidden_state, mask_actions)
				
					one_hot_actions = np.zeros((self.num_agents, self.num_actions))
					for i,act in enumerate(actions):
						one_hot_actions[i][act] = 1

					self.buffer.end_episode(full_state, one_hot_actions, dones)

					break

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
				torch.save(self.agents.actor.state_dict(), self.model_path+'_actor_epsiode'+str(episode)+'.pt')
				torch.save(self.agents.critic.state_dict(), self.model_path+'_critic_epsiode'+str(episode)+'.pt')

			if self.learn and episode != 0 and episode%self.update_episode_interval == 0:
				self.agents.update(self.buffer, episode)
				self.buffer.clear()

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
				



if __name__ == '__main__':

	RENDER = False
	USE_CPP_RVO2 = False

	torch.set_printoptions(profile="full")
	torch.autograd.set_detect_anomaly(True)

	for i in range(1,4):
		extension = "LICA_"+str(i)
		test_num = "StarCraft"
		environment = "LBForaging" # StarCraft/ MPE/ PressurePlate/ PettingZoo/ LBForaging
		if "LBForaging" in environment:
			num_players = 6
			num_food = 9
			grid_size = 12
			fully_coop = False
		env_name = "Foraging-{0}x{0}-{1}p-{2}f{3}-v2".format(grid_size, num_players, num_food, "-coop" if fully_coop else "") # 5m_vs_6m/ 10m_vs_11m/ 3s5z/ crossing_team_greedy/ pressureplate-linear-6p-v0/ pursuit_v4/ "Foraging-{0}x{0}-{1}p-{2}f{3}-v2".format(grid_size, num_players, num_food, "-coop" if fully_coop else "")

		dictionary = {
				# TRAINING
				"environment": environment,
				"iteration": i,
				"device": "gpu",
				"model_dir": '../../../tests/'+test_num+'/models/'+env_name+'_'+'_'+extension+'/models/',
				"gif_dir": '../../../tests/'+test_num+'/gifs/'+env_name+'_'+'_'+extension+'/',
				"policy_eval_dir":'../../../tests/'+test_num+'/policy_eval/'+env_name+'_'+'_'+extension+'/',
				"test_num":test_num,
				"extension":extension,
				"gamma": 0.99,
				"gif": False,
				"gif_checkpoint":1,
				"load_models": False,
				"model_path_actor": "../../../tests/PRD_2_MPE/models/crossing_team_greedy_prd_above_threshold_MAPPO_Q_run_2/critic_networks/critic_epsiode100000.pt",
				"model_path_critic": "../../../tests/PRD_2_MPE/models/crossing_team_greedy_prd_above_threshold_MAPPO_Q_run_2/critic_networks/critic_epsiode100000.pt",
				"eval_policy": True,
				"save_model": True,
				"save_model_checkpoint": 1000,
				"save_comet_ml_plot": True,
				"norm_returns": False,
				"learn":True,
				"max_episodes": 15000,
				"max_time_steps": 70,
				"parallel_training": False,
				"scheduler_need": False,
				"update_episode_interval": 10,
				"num_updates": 1,
				"entropy_coeff": 1e-2,
				"lambda": 0.8,

				# ENVIRONMENT
				"env": env_name,

				# MODEL
				"critic_learning_rate": 5e-4, #1e-3
				"actor_learning_rate": 5e-4, #1e-3
				"actor_weight_decay": 0.0,
				"critic_weight_decay": 0.0,
				"enable_grad_clip_critic": True,
				"enable_grad_clip_actor": True,
				"critic_grad_clip": 10.0,
				"actor_grad_clip": 10.0,
				"rnn_num_layers": 1,
				"rnn_hidden_dim": 64,
				"data_chunk_length": 10,
				"mixing_embed_dim": 64,
				"num_hypernet_layers": 2,
				"norm_returns": False,
				"soft_update": False,
				"tau": 0.001,
				"target_update_interval": 200,
			}

		seeds = [42, 142, 242, 342, 442]
		torch.manual_seed(seeds[dictionary["iteration"]-1])

		if "StarCraft" in environment:

			import gym
			import smaclite  # noqa


			env = gym.make(f"smaclite/{env_name}-v0", use_cpp_rvo2=USE_CPP_RVO2)
			obs, info = env.reset(return_info=True)
			dictionary["local_observation"] = obs[0].shape[0]+env.n_agents
			dictionary["global_observation"] = info["ally_states"].reshape(-1).shape[0] + info["enemy_states"].reshape(-1).shape[0] + env.n_agents**2
		
		elif "MPE" in environment:

			from multiagent.environment import MultiAgentEnv
			import multiagent.scenarios as scenarios

			env, local_observation, global_observation = make_env(scenario_name=dictionary["env"], benchmark=False)
			dictionary["global_observation"] = global_observation
			dictionary["local_observation"] = q_observation_shape+env.action_space[0].n
			dictionary["num_agents"] = env.n
			dictionary["num_actions"] = env.action_space[0].n

		elif "PressurePlate" in environment:

			import pressureplate
			import gym

			env = gym.make(env_name)
			dictionary["global_observation"] = (133+env.n_agents)*env.n_agents
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
			
			dictionary["global_observation"] = (obs_range*obs_range*3 + num_agents)*num_agents
			dictionary["local_observation"] = obs_range*obs_range*3 + num_agents + num_actions
			dictionary["num_agents"] = num_agents
			dictionary["num_actions"] = num_actions

		elif "LBForaging" in environment:

			import lbforaging
			import gym

			env = gym.make(env_name, max_episode_steps=dictionary["max_time_steps"], penalty=0.0, normalize_reward=True)
			dictionary["global_observation"] = (num_players*3 + num_food*3 + num_players) * num_players
			dictionary["local_observation"] = num_players*3 + num_food*3 + num_players + env.action_space[0].n
			dictionary["num_agents"] = num_players
			dictionary["num_actions"] = env.action_space[0].n


		ma_controller = LICA(env, dictionary)
		ma_controller.run()
