from multiagent.environment import MultiAgentEnv
import multiagent.scenarios as scenarios

import os
import time
from comet_ml import Experiment
import numpy as np
from agent import QMIXAgent
from buffer import ReplayMemory
import torch
import datetime

torch.autograd.set_detect_anomaly(True)



class QMIX:

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
		self.num_agents = self.env.n
		self.num_actions = self.env.action_space[0].n
		self.date_time = f"{datetime.datetime.now():%d-%m-%Y}"
		self.env_name = dictionary["env"]
		self.test_num = dictionary["test_num"]
		self.max_episodes = dictionary["max_episodes"]
		self.max_time_steps = dictionary["max_time_steps"]

		self.observation_shape = dictionary["observation_shape"]
		self.replay_buffer_size = dictionary["replay_buffer_size"]
		self.batch_size = dictionary["batch_size"] # number of datapoints to sample
		self.buffer = ReplayMemory(
			capacity = self.replay_buffer_size,
			max_episode_len = self.max_time_steps,
			num_agents = self.num_agents,
			obs_shape = self.observation_shape,
			action_shape = self.num_actions
			)

		self.epsilon_greedy = dictionary["epsilon_greedy"]
		self.epsilon_greedy_min = dictionary["epsilon_greedy_min"]
		self.epsilon_decay_rate = (self.epsilon_greedy - self.epsilon_greedy_min) / dictionary["epsilon_greedy_decay_episodes"]


		self.comet_ml = None
		if self.save_comet_ml_plot:
			self.comet_ml = Experiment("im5zK8gFkz6j07uflhc3hXk8I",project_name=dictionary["test_num"])
			self.comet_ml.log_parameters(dictionary)


		self.agents = QMIXAgent(self.env, dictionary, self.comet_ml)
		# self.init_critic_hidden_state(np.zeros((1, self.num_agents, 256)))

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


	def split_states(self,states):
		states_critic = []
		states_actor = []
		for i in range(self.num_agents):
			states_critic.append(states[i][0])
			states_actor.append(states[i][1])

		states_critic = np.asarray(states_critic)
		states_actor = np.asarray(states_actor)

		return states_critic, states_actor


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
			self.collision_rates = []
			self.collison_rate_mean_per_1000_eps = []

		for episode in range(self.max_episodes):

			states = self.env.reset()

			images = []

			_, states = self.split_states(states)

			episode_reward = 0
			episode_collision_rate = 0
			episode_goal_reached = 0
			final_timestep = self.max_time_steps

			last_one_hot_action = np.zeros((self.num_agents, self.num_actions))
			self.agents.Q_network.rnn_hidden_state = None
			self.agents.target_Q_network.rnn_hidden_state = None

			for step in range(1, self.max_time_steps+1):

				if self.gif:
					# At each step, append an image to list
					if not(episode%self.gif_checkpoint):
						images.append(np.squeeze(self.env.render(mode='rgb_array')))
					import time
					# time.sleep(0.1)
					# Advance a step and render a new image
					with torch.no_grad():
						actions = self.agents.get_action(states, last_one_hot_action)
				else:
					actions = self.agents.get_action(states, last_one_hot_action)

				last_one_hot_action = np.zeros((self.num_agents,self.num_actions))
				for i,act in enumerate(actions):
					last_one_hot_action[i][act] = 1

				next_states, rewards, dones, info = self.env.step(actions)
				_, next_states = self.split_states(next_states)

				if "crossing" in self.env_name:
					collision_rate = [value[1] for value in rewards]
					goal_reached = [value[2] for value in rewards]
					rewards = [value[0] for value in rewards] 
					episode_collision_rate += np.sum(collision_rate)
					episode_goal_reached += np.sum(goal_reached)

				if not self.gif:
					self.buffer.push(states, actions, last_one_hot_action, np.sum(rewards), all(dones))

				episode_reward += np.sum(rewards)

				states = next_states

				if all(dones) or step == self.max_time_steps:

					print("*"*100)
					print("EPISODE: {} | REWARD: {} | TIME TAKEN: {} / {} \n".format(episode,np.round(episode_reward,decimals=4),step,self.max_time_steps))
					print("*"*100)

					final_timestep = step

					if self.save_comet_ml_plot:
						self.comet_ml.log_metric('Episode_Length', step, episode)
						self.comet_ml.log_metric('Reward', episode_reward, episode)
						self.comet_ml.log_metric('Number of Collision', episode_collision_rate, episode)
						self.comet_ml.log_metric('Num Agents Goal Reached', np.sum(dones), episode)

					break

			self.epsilon_greedy = self.epsilon_greedy - self.epsilon_decay_rate if self.epsilon_greedy - self.epsilon_decay_rate > self.epsilon_greedy_min else self.epsilon_greedy_min
			self.buffer.end_episode(states)

			if self.eval_policy:
				self.rewards.append(episode_reward)
				self.timesteps.append(final_timestep)
				self.collision_rates.append(episode_collision_rate)

			if episode > self.save_model_checkpoint and self.eval_policy:
				self.rewards_mean_per_1000_eps.append(sum(self.rewards[episode-self.save_model_checkpoint:episode])/self.save_model_checkpoint)
				self.timesteps_mean_per_1000_eps.append(sum(self.timesteps[episode-self.save_model_checkpoint:episode])/self.save_model_checkpoint)
				self.collison_rate_mean_per_1000_eps.append(sum(self.collision_rates[episode-self.save_model_checkpoint:episode])/self.save_model_checkpoint)


			if not(episode%self.save_model_checkpoint) and episode!=0 and self.save_model:	
				torch.save(self.agents.Q_network.state_dict(), self.model_path+'_Q_epsiode'+str(episode)+'.pt')
				torch.save(self.agents.QMix_network.state_dict(), self.model_path+'_QMix_epsiode'+str(episode)+'.pt')

			if self.learn and self.batch_size <= self.buffer.length and episode != 0:
				sample = self.buffer.sample(num_episodes=self.batch_size)
				self.agents.update(sample, episode)

			elif self.gif and not(episode%self.gif_checkpoint):
				print("GENERATING GIF")
				self.make_gif(np.array(images),self.gif_path)


			if self.eval_policy and not(episode%self.save_model_checkpoint) and episode!=0:
				np.save(os.path.join(self.policy_eval_dir,self.test_num+"reward_list"), np.array(self.rewards), allow_pickle=True, fix_imports=True)
				np.save(os.path.join(self.policy_eval_dir,self.test_num+"mean_rewards_per_1000_eps"), np.array(self.rewards_mean_per_1000_eps), allow_pickle=True, fix_imports=True)
				np.save(os.path.join(self.policy_eval_dir,self.test_num+"timestep_list"), np.array(self.timesteps), allow_pickle=True, fix_imports=True)
				np.save(os.path.join(self.policy_eval_dir,self.test_num+"mean_timestep_per_1000_eps"), np.array(self.timesteps_mean_per_1000_eps), allow_pickle=True, fix_imports=True)
				np.save(os.path.join(self.policy_eval_dir,self.test_num+"collision_rate_list"), np.array(self.collision_rates), allow_pickle=True, fix_imports=True)
				np.save(os.path.join(self.policy_eval_dir,self.test_num+"mean_collision_rate_per_1000_eps"), np.array(self.collison_rate_mean_per_1000_eps), allow_pickle=True, fix_imports=True)


def make_env(scenario_name, benchmark=False):
	scenario = scenarios.load(scenario_name + ".py").Scenario()
	world = scenario.make_world()
	if benchmark:
		env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation, scenario.benchmark_data, scenario.isFinished)
	else:
		env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation, None, scenario.isFinished)
	return env, scenario.observation_shape


if __name__ == '__main__':

	for i in range(1,6):
		extension = "QMix"+str(i)
		test_num = "TEAM COLLISION AVOIDANCE"
		env_name = "crossing_team_greedy"

		dictionary = {
				# TRAINING
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
				"model_path": "../../../tests/PRD_2_MPE/models/crossing_team_greedy_prd_above_threshold_MAPPO_Q_run_2/critic_networks/critic_epsiode100000.pt",
				"eval_policy": True,
				"save_model": True,
				"save_model_checkpoint": 1000,
				"save_comet_ml_plot": True,
				"learn":True,
				"max_episodes": 50000,
				"max_time_steps": 100,
				"parallel_training": False,
				"scheduler_need": False,
				"replay_buffer_size": 50000,
				"batch_size": 500,
				"epsilon_greedy": 1.0,
				"epsilon_greedy_min": 0.05,
				"epsilon_greedy_decay_episodes": 5000,

				# ENVIRONMENT
				"env": env_name,

				# MODEL
				"learning_rate": 1e-5, #1e-3
				"grad_clip": 10.0,
				"rnn_hidden_dim": 64,
				"hidden_dim": 32,
				"norm_returns": False,
				"soft_update": False,
				"tau": 0.005,
				"target_update_interval": 200,
			}

		seeds = [42, 142, 242, 342, 442]
		torch.manual_seed(seeds[dictionary["iteration"]-1])
		env, observation_shape = make_env(scenario_name=dictionary["env"], benchmark=False)
		dictionary["observation_shape"] = observation_shape
		ma_controller = QMIX(env, dictionary)
		ma_controller.run()
