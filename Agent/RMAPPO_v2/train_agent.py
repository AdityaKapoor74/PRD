import os
import time
from comet_ml import Experiment
import numpy as np
from agent import PPOAgent
import torch
import datetime

import gym
import smaclite  # noqa

torch.autograd.set_detect_anomaly(True)



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
		self.num_enemies = self.env.n_enemies
		self.num_actions = self.env.action_space[0].n
		self.date_time = f"{datetime.datetime.now():%d-%m-%Y}"
		self.env_name = dictionary["env"]
		self.test_num = dictionary["test_num"]
		self.max_episodes = dictionary["max_episodes"]
		self.max_time_steps = dictionary["max_time_steps"]
		self.experiment_type = dictionary["experiment_type"]
		self.update_ppo_agent = dictionary["update_ppo_agent"]

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

		self.enemy_ids = []
		for i in range(self.num_enemies):
			enemy_id = np.array([0 for i in range(self.num_enemies)])
			enemy_id[i] = 1
			self.enemy_ids.append(enemy_id)
		self.enemy_ids = np.array(self.enemy_ids)


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




	def run(self):  
		if self.eval_policy:
			self.rewards = []
			self.rewards_mean_per_1000_eps = []
			self.timesteps = []
			self.timesteps_mean_per_1000_eps = []

		# init_rnn_hidden_v = torch.zeros(self.num_agents, self.rnn_hidden_v)
		# init_rnn_hidden_q = torch.zeros(self.num_agents, self.rnn_hidden_q)
		# init_rnn_hidden_actor = torch.zeros(self.num_agents, self.rnn_hidden_actor)

		for episode in range(1,self.max_episodes+1):

			states_actor, info = self.env.reset(return_info=True)
			mask_actions = np.array(info["avail_actions"], dtype=int)
			last_one_hot_actions = np.zeros((self.num_agents, self.num_actions))
			states_allies_critic = np.concatenate((self.agent_ids, info["ally_states"]), axis=-1)
			states_enemies_critic = np.concatenate((self.enemy_ids, info["enemy_states"]), axis=-1)
			states_actor = np.array(states_actor)
			states_actor = np.concatenate((self.agent_ids, states_actor), axis=-1)
			indiv_dones = [0]*self.num_agents
			indiv_dones = np.array(indiv_dones)
			

			images = []

			episode_reward = 0
			final_timestep = self.max_time_steps

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
						actions, action_logprob, next_rnn_hidden_state_actor = self.agents.get_action(states_actor, last_one_hot_actions, mask_actions, rnn_hidden_state_actor, greedy=False)
				else:
					actions, action_logprob, next_rnn_hidden_state_actor = self.agents.get_action(states_actor, last_one_hot_actions, mask_actions, rnn_hidden_state_actor)

				one_hot_actions = np.zeros((self.num_agents, self.num_actions))
				for i, act in enumerate(actions):
					one_hot_actions[i][act] = 1

				q_value, next_rnn_hidden_state_q, weights_prd, value, next_rnn_hidden_state_v = self.agents.get_q_v_values(states_allies_critic, states_enemies_critic, one_hot_actions, rnn_hidden_state_q, rnn_hidden_state_v)


				next_states_actor, rewards, next_dones, info = self.env.step(actions)
				next_states_actor = np.array(next_states_actor)
				next_states_actor = np.concatenate((self.agent_ids, next_states_actor), axis=-1)
				next_states_allies_critic = np.concatenate((self.agent_ids, info["ally_states"]), axis=-1)
				next_states_enemies_critic = np.concatenate((self.enemy_ids, info["enemy_states"]), axis=-1)
				next_mask_actions = np.array(info["avail_actions"], dtype=int)
				next_indiv_dones = info["indiv_dones"]

				if self.learn:
					if self.experiment_type == "shared":
						rewards_to_send = [rewards]*self.num_agents
						# rewards_to_send = [rewards if indiv_dones[i]==0 else 0 for i in range(self.num_agents)]
					else:
						rewards_to_send = info["indiv_rewards"]

					self.agents.buffer.push(
						states_allies_critic, states_enemies_critic, q_value, rnn_hidden_state_q, weights_prd, value, rnn_hidden_state_v, \
						states_actor, rnn_hidden_state_actor, action_logprob, actions, last_one_hot_actions, one_hot_actions, mask_actions, \
						rewards_to_send, indiv_dones
						)

				episode_reward += np.sum(rewards)

				states_actor, last_one_hot_actions, states_allies_critic, states_enemies_critic, mask_actions, indiv_dones = next_states_actor, one_hot_actions, next_states_allies_critic, next_states_enemies_critic, next_mask_actions, next_indiv_dones
				rnn_hidden_state_q, rnn_hidden_state_v, rnn_hidden_state_actor = next_rnn_hidden_state_q, next_rnn_hidden_state_v, next_rnn_hidden_state_actor

				if all(indiv_dones) or step == self.max_time_steps:

					print("*"*100)
					print("EPISODE: {} | REWARD: {} | TIME TAKEN: {} / {} | Num Allies Alive: {} | Num Enemies Alive: {} \n".format(episode, np.round(episode_reward,decimals=4), step, self.max_time_steps, info["num_allies"], info["num_enemies"]))
					print("*"*100)

					final_timestep = step

					if self.save_comet_ml_plot:
						self.comet_ml.log_metric('Episode_Length', step, episode)
						self.comet_ml.log_metric('Reward', episode_reward, episode)
						self.comet_ml.log_metric('Num Enemies', info["num_enemies"], episode)
						self.comet_ml.log_metric('Num Allies', info["num_allies"], episode)
						self.comet_ml.log_metric('All Enemies Dead', info["all_enemies_dead"], episode)
						self.comet_ml.log_metric('All Allies Dead', info["all_allies_dead"], episode)

					# if warmup
					# self.agents.update_epsilon()

					if self.learn:
						# add final time to buffer
						actions, action_logprob, next_rnn_hidden_state_actor = self.agents.get_action(states_actor, last_one_hot_actions, mask_actions, rnn_hidden_state_actor)
					
						one_hot_actions = np.zeros((self.num_agents,self.num_actions))
						for i,act in enumerate(actions):
							one_hot_actions[i][act] = 1

						q_value, _, _, value, _ = self.agents.get_q_v_values(states_allies_critic, states_enemies_critic, one_hot_actions, rnn_hidden_state_q, rnn_hidden_state_v)

						self.agents.buffer.end_episode(final_timestep, q_value, value, indiv_dones)

					break

			if self.agents.scheduler_need:
				self.agents.scheduler_policy.step()
				self.agents.scheduler_q_critic.step()
				self.agents.scheduler_v_critic.step()

			if self.eval_policy:
				self.rewards.append(episode_reward)
				self.timesteps.append(final_timestep)

			if episode > self.save_model_checkpoint and self.eval_policy:
				self.rewards_mean_per_1000_eps.append(sum(self.rewards[episode-self.save_model_checkpoint:episode])/self.save_model_checkpoint)
				self.timesteps_mean_per_1000_eps.append(sum(self.timesteps[episode-self.save_model_checkpoint:episode])/self.save_model_checkpoint)

			if not(episode%self.save_model_checkpoint) and episode!=0 and self.save_model:	
				torch.save(self.agents.critic_network_q.state_dict(), self.critic_model_path+'_Q_epsiode'+str(episode)+'.pt')
				torch.save(self.agents.critic_network_v.state_dict(), self.critic_model_path+'_V_epsiode'+str(episode)+'.pt')
				torch.save(self.agents.policy_network.state_dict(), self.actor_model_path+'_epsiode'+str(episode)+'.pt')  

			if self.learn and not(episode%self.update_ppo_agent) and episode != 0:
				self.agents.update(episode)

			# elif self.gif and not(episode%self.gif_checkpoint):
			# 	print("GENERATING GIF")
			# 	self.make_gif(np.array(images),self.gif_path)


			if self.eval_policy and not(episode%self.save_model_checkpoint) and episode!=0:
				np.save(os.path.join(self.policy_eval_dir,self.test_num+"reward_list"), np.array(self.rewards), allow_pickle=True, fix_imports=True)
				np.save(os.path.join(self.policy_eval_dir,self.test_num+"mean_rewards_per_1000_eps"), np.array(self.rewards_mean_per_1000_eps), allow_pickle=True, fix_imports=True)
				np.save(os.path.join(self.policy_eval_dir,self.test_num+"timestep_list"), np.array(self.timesteps), allow_pickle=True, fix_imports=True)
				np.save(os.path.join(self.policy_eval_dir,self.test_num+"mean_timestep_per_1000_eps"), np.array(self.timesteps_mean_per_1000_eps), allow_pickle=True, fix_imports=True)
				

if __name__ == '__main__':

	RENDER = False
	USE_CPP_RVO2 = False

	for i in range(1,4):
		extension = "MAPPO_"+str(i)
		test_num = "StarCraft"
		env_name = "5m_vs_6m"
		experiment_type = "prd_above_threshold_ascend" # shared, prd_above_threshold_ascend, prd_above_threshold, prd_top_k, prd_above_threshold_decay, prd_soft_advantage

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
				"update_ppo_agent": 10, # update ppo agent after every update_ppo_agent episodes
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
				"max_episodes": 20000,
				"max_time_steps": 100,
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
				"enable_grad_clip_critic_v": False,
				"grad_clip_critic_v": 2.0,
				"enable_grad_clip_critic_q": False,
				"grad_clip_critic_q": 0.2,
				"value_clip": 0.05,
				"enable_hard_attention": False,
				"num_heads": 1,
				"critic_weight_entropy_pen": 0.0,
				"critic_weight_entropy_pen_final": 0.0,
				"critic_weight_entropy_pen_steps": 100, # number of updates
				"critic_score_regularizer": 0.0,
				"target_calc_style": "GAE", # GAE, TD_Lambda, N_steps
				"td_lambda": 0.95, # 1 --> Monte Carlo; 0 --> TD(1)
				"n_steps": 5,
				"norm_returns_q": True,
				"norm_returns_v": True,
				

				# ACTOR
				"data_chunk_length": 10,
				"rnn_num_layers_actor": 1,
				"rnn_hidden_actor": 64,
				"enable_grad_clip_actor": False,
				"grad_clip_actor": 0.2,
				"policy_clip": 0.02,
				"policy_lr": 5e-4, #prd 1e-4
				"policy_weight_decay": 0.0,
				"entropy_pen": 1e-3, #8e-3
				"entropy_pen_final": 1e-3,
				"entropy_pen_steps": 20000,
				"gae_lambda": 0.95,
				"select_above_threshold": 0.0, #0.043, 0.1
				"threshold_min": 0.0, 
				"threshold_max": 0.25, #0.12
				"steps_to_take": 1000,
				"top_k": 0,
				"norm_adv": True,

				"soft_update_q": False,
				"tau_q": 0.05,
				"network_update_interval_q": 1,
				"soft_update_v": False,
				"tau_v": 0.05,
				"network_update_interval_v": 1,
			}

		seeds = [42, 142, 242, 342, 442]
		torch.manual_seed(seeds[dictionary["iteration"]-1])
		env = gym.make(f"smaclite/{env_name}-v0", use_cpp_rvo2=USE_CPP_RVO2)
		obs, info = env.reset(return_info=True)
		dictionary["ally_observation"] = info["ally_states"][0].shape[0]+env.n_agents #+env.action_space[0].n #4+env.action_space[0].n+env.n_agents
		dictionary["enemy_observation"] = info["enemy_states"][0].shape[0]+env.n_enemies
		dictionary["local_observation"] = obs[0].shape[0]+env.n_agents
		ma_controller = MAPPO(env,dictionary)
		ma_controller.run()
