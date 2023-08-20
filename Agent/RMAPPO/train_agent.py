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
		self.num_actions = self.env.action_space[0].n
		self.date_time = f"{datetime.datetime.now():%d-%m-%Y}"
		self.env_name = dictionary["env"]
		self.test_num = dictionary["test_num"]
		self.max_episodes = dictionary["max_episodes"]
		self.max_time_steps = dictionary["max_time_steps"]
		self.experiment_type = dictionary["experiment_type"]
		self.update_ppo_agent = dictionary["update_ppo_agent"]

		# RNN HIDDEN
		self.rnn_hidden_q = dictionary["rnn_hidden_q"]
		self.rnn_hidden_v = dictionary["rnn_hidden_v"]
		self.rnn_hidden_actor = dictionary["rnn_hidden_actor"]

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
			# concatenate state information with last action and agent id
			states_allies_critic = np.concatenate((self.agent_ids, info["ally_states"], np.zeros((self.num_agents, self.num_actions))), axis=-1)
			states_enemies_critic = info["enemy_states"]
			states_actor = np.array(states_actor)
			states_actor = np.concatenate((self.agent_ids, states_actor), axis=-1)

			images = []

			episode_reward = 0
			final_timestep = self.max_time_steps

			# self.agents.policy_network_old.rnn_hidden_state = None
			# self.agents.policy_network.rnn_hidden_state = init_rnn_hidden_actor.to(self.device)

			# self.agents.critic_network_q_old.rnn_hidden_state = None
			# self.agents.critic_network_q.rnn_hidden_state = init_rnn_hidden_q.to(self.device)

			# self.agents.critic_network_v_old.rnn_hidden_state = None
			# self.agents.critic_network_v.rnn_hidden_state = init_rnn_hidden_v.to(self.device)

			# rnn_hidden_state_q = np.zeros((1, self.num_agents, self.rnn_hidden_q))
			# rnn_hidden_state_v = np.zeros((1, self.num_agents, self.rnn_hidden_v))
			rnn_hidden_state_actor = np.zeros((1, self.num_agents, self.rnn_hidden_actor))

			for step in range(1, self.max_time_steps+1):

				if self.gif:
					# At each step, append an image to list
					if not(episode%self.gif_checkpoint):
						images.append(np.squeeze(self.env.render(mode='rgb_array')))
					import time
					time.sleep(0.1)
					# Advance a step and render a new image
					with torch.no_grad():
						actions, _, rnn_hidden_state_actor = self.agents.get_action(states_actor, mask_actions, rnn_hidden_state_actor, greedy=True)
				else:
					actions, action_logprob, rnn_hidden_state_actor = self.agents.get_action(states_actor, mask_actions, rnn_hidden_state_actor)
					one_hot_actions = np.zeros((self.num_agents,self.num_actions))
					for i,act in enumerate(actions):
						one_hot_actions[i][act] = 1

					# rnn_hidden_state_q, rnn_hidden_state_v = self.agents.get_critic_hidden_state(states_allies_critic, states_enemies_critic, one_hot_actions, rnn_hidden_state_q, rnn_hidden_state_v)


				next_states_actor, rewards, dones, info = self.env.step(actions)
				dones = [int(dones)]*self.num_agents
				rewards = info["indiv_rewards"]
				next_states_actor = np.array(next_states_actor)
				next_states_actor = np.concatenate((self.agent_ids, next_states_actor), axis=-1)
				next_states_allies_critic = np.concatenate((self.agent_ids, info["ally_states"], one_hot_actions), axis=-1)
				next_states_enemies_critic = info["enemy_states"]
				next_mask_actions = np.array(info["avail_actions"], dtype=int)

				if not self.gif:
					self.agents.buffer.push(states_allies_critic, states_enemies_critic, states_actor, action_logprob, actions, one_hot_actions, mask_actions, rewards, dones)

				episode_reward += np.sum(rewards)

				states_actor, states_allies_critic, states_enemies_critic, mask_actions = next_states_actor, next_states_allies_critic, next_states_enemies_critic, next_mask_actions

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

					# if warmup
					# self.agents.update_epsilon()

					# add final time to buffer
					self.agents.buffer.end_episode(final_timestep)

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

			elif self.gif and not(episode%self.gif_checkpoint):
				print("GENERATING GIF")
				self.make_gif(np.array(images),self.gif_path)


			if self.eval_policy and not(episode%self.save_model_checkpoint) and episode!=0:
				np.save(os.path.join(self.policy_eval_dir,self.test_num+"reward_list"), np.array(self.rewards), allow_pickle=True, fix_imports=True)
				np.save(os.path.join(self.policy_eval_dir,self.test_num+"mean_rewards_per_1000_eps"), np.array(self.rewards_mean_per_1000_eps), allow_pickle=True, fix_imports=True)
				np.save(os.path.join(self.policy_eval_dir,self.test_num+"timestep_list"), np.array(self.timesteps), allow_pickle=True, fix_imports=True)
				np.save(os.path.join(self.policy_eval_dir,self.test_num+"mean_timestep_per_1000_eps"), np.array(self.timesteps_mean_per_1000_eps), allow_pickle=True, fix_imports=True)
				

if __name__ == '__main__':

	RENDER = False
	USE_CPP_RVO2 = False

	for i in range(1,6):
		extension = "MAPPO_"+str(i)
		test_num = "StarCraft"
		env_name = "10m_vs_11m"
		experiment_type = "shared" # shared, prd_above_threshold_ascend, prd_above_threshold, prd_top_k, prd_above_threshold_decay, prd_soft_advantage

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
				"update_ppo_agent": 5, # update ppo agent after every update_ppo_agent episodes
				"test_num":test_num,
				"extension":extension,
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
				"epsilon_start": 1.0,
				"epsilon_end": 0.0,
				"max_episodes": 30000,
				"max_time_steps": 25,
				"experiment_type": experiment_type,
				"parallel_training": False,
				"scheduler_need": False,


				# ENVIRONMENT
				"env": env_name,

				# CRITIC
				"rnn_hidden_q": 64,
				"rnn_hidden_v": 64,				
				"q_value_lr": 1e-3, #1e-3
				"v_value_lr": 1e-3, #1e-3
				"temperature_v": 1.0,
				"temperature_q": 1.0,
				"attention_dropout_prob_q": 0.0,
				"attention_dropout_prob_v": 0.0,
				"q_weight_decay": 5e-4,
				"v_weight_decay": 5e-4,
				"enable_grad_clip_critic": False,
				"grad_clip_critic": 10.0,
				"value_clip": 0.2,
				"enable_hard_attention": False,
				"num_heads": 4,
				"critic_weight_entropy_pen": 2e-3,
				"critic_weight_entropy_pen_final": 0.0,
				"critic_weight_entropy_pen_steps": 100, # number of updates
				"critic_score_regularizer": 0.0,
				"lambda": 0.95, # 1 --> Monte Carlo; 0 --> TD(1)
				"norm_returns": False,
				

				# ACTOR
				"rnn_hidden_actor": 64,
				"enable_grad_clip_actor": False,
				"grad_clip_actor": 10.0,
				"policy_clip": 0.2,
				"policy_lr": 8e-3, #prd 1e-4
				"policy_weight_decay": 5e-4,
				"entropy_pen": 1e-3, #8e-3
				"gae_lambda": 0.95,
				"select_above_threshold": 0.1,
				"threshold_min": 0.0, 
				"threshold_max": 0.0, #0.12
				"steps_to_take": 1000,
				"top_k": 0,
				"norm_adv": False,

				"network_update_interval": 1,
			}

		seeds = [42, 142, 242, 342, 442]
		torch.manual_seed(seeds[dictionary["iteration"]-1])
		env = gym.make(f"smaclite/{env_name}-v0", use_cpp_rvo2=USE_CPP_RVO2)
		obs, info = env.reset(return_info=True)
		dictionary["ally_observation"] = 4+env.action_space[0].n+env.n_agents
		dictionary["enemy_observation"] = 3
		dictionary["local_observation"] = obs[0].shape[0]+env.n_agents
		ma_controller = MAPPO(env,dictionary)
		ma_controller.run()
