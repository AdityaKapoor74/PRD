from multiagent.environment import MultiAgentEnv
import multiagent.scenarios as scenarios

import os
import time
from comet_ml import Experiment
import numpy as np
from agent import PPOAgent
import torch
import datetime

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
		self.num_agents = self.env.n
		self.num_actions = self.env.action_space[0].n
		self.date_time = f"{datetime.datetime.now():%d-%m-%Y}"
		self.env_name = dictionary["env"]
		self.test_num = dictionary["test_num"]
		self.max_episodes = dictionary["max_episodes"]
		self.max_time_steps = dictionary["max_time_steps"]
		self.experiment_type = dictionary["experiment_type"]
		self.update_ppo_agent = dictionary["update_ppo_agent"]


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


	def split_states(self,states):
		states_critic = []
		states_actor = []
		for i in range(self.num_agents):
			states_critic.append(states[i][0])
			states_actor.append(states[i][1])

		states_critic = np.asarray(states_critic)
		states_actor = np.asarray(states_actor)

		return states_critic,states_actor

	# def init_critic_hidden_state(self, hidden_state):
	# 	self.critic_hidden_state = hidden_state


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

		for episode in range(1,self.max_episodes+1):

			states = self.env.reset()

			images = []

			states_critic, states_actor = self.split_states(states)

			episode_reward = 0
			episode_collision_rate = 0
			episode_goal_reached = 0
			final_timestep = self.max_time_steps
			for step in range(1, self.max_time_steps+1):

				if self.gif:
					# At each step, append an image to list
					if not(episode%self.gif_checkpoint):
						images.append(np.squeeze(self.env.render(mode='rgb_array')))
					import time
					time.sleep(0.1)
					# Advance a step and render a new image
					with torch.no_grad():
						actions = self.agents.get_action(states_actor, greedy=True)
				else:
					actions = self.agents.get_action(states_actor)
					one_hot_actions = np.zeros((self.num_agents,self.num_actions))
					for i,act in enumerate(actions):
						one_hot_actions[i][act] = 1

					# Q_value, V_value, critic_hidden_state, _ = self.agents.critic_network_old(
					# 															torch.FloatTensor(np.array(states_critic)).unsqueeze(0).to(self.device),
					# 															torch.from_numpy(self.critic_hidden_state).double().unsqueeze(0).to(self.device),
					# 															torch.FloatTensor(np.array(one_hot_actions)).long().unsqueeze(0).to(self.device)
					# 															)

				

				next_states, rewards, dones, info = self.env.step(actions)
				next_states_critic, next_states_actor = self.split_states(next_states)

				if "crossing" in self.env_name:
					collision_rate = [value[1] for value in rewards]
					goal_reached = [value[2] for value in rewards]
					rewards = [value[0] for value in rewards] 
					episode_collision_rate += np.sum(collision_rate)
					episode_goal_reached += np.sum(goal_reached)


				# if step == self.max_time_steps:
				# 	dones = [True for _ in range(self.num_agents)]

				if not self.gif:
					self.agents.buffer.states_critic.append(states_critic)
					# self.agents.buffer.history_states_critic.append(self.critic_hidden_state)
					# self.agents.buffer.Q_values.append(Q_value.squeeze(0).detach().cpu())
					# self.agents.buffer.Values.append(V_value.squeeze(0).detach().cpu())
					self.agents.buffer.states_actor.append(states_actor)
					self.agents.buffer.actions.append(actions)
					self.agents.buffer.one_hot_actions.append(one_hot_actions)
					self.agents.buffer.dones.append(dones)
					self.agents.buffer.rewards.append(rewards)

					self.init_critic_hidden_state(critic_hidden_state.detach().cpu().numpy())

				episode_reward += np.sum(rewards)

				states_critic, states_actor = next_states_critic, next_states_actor
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

			if self.eval_policy:
				self.rewards.append(episode_reward)
				self.timesteps.append(final_timestep)
				self.collision_rates.append(episode_collision_rate)

			if episode > self.save_model_checkpoint and self.eval_policy:
				self.rewards_mean_per_1000_eps.append(sum(self.rewards[episode-self.save_model_checkpoint:episode])/self.save_model_checkpoint)
				self.timesteps_mean_per_1000_eps.append(sum(self.timesteps[episode-self.save_model_checkpoint:episode])/self.save_model_checkpoint)
				self.collison_rate_mean_per_1000_eps.append(sum(self.collision_rates[episode-self.save_model_checkpoint:episode])/self.save_model_checkpoint)


			if not(episode%self.save_model_checkpoint) and episode!=0 and self.save_model:	
				torch.save(self.agents.critic_network.state_dict(), self.critic_model_path+'_epsiode'+str(episode)+'.pt')
				torch.save(self.agents.policy_network.state_dict(), self.actor_model_path+'_epsiode'+str(episode)+'.pt')  

			if self.learn and not(episode%self.update_ppo_agent) and episode != 0:
				self.agents.update(episode)
				# history_states_critic = self.agents.update(episode)
				# self.init_critic_hidden_state(history_states_critic.detach().unsqueeze(0).cpu().numpy())
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

				if "prd" in self.experiment_type:
					np.save(os.path.join(self.policy_eval_dir,self.test_num+"num_relevant_agents_in_relevant_set"), np.array(self.agents.num_relevant_agents_in_relevant_set), allow_pickle=True, fix_imports=True)
					np.save(os.path.join(self.policy_eval_dir,self.test_num+"num_non_relevant_agents_in_relevant_set"), np.array(self.agents.num_non_relevant_agents_in_relevant_set), allow_pickle=True, fix_imports=True)
					np.save(os.path.join(self.policy_eval_dir,self.test_num+"false_positive_rate"), np.array(self.agents.false_positive_rate), allow_pickle=True, fix_imports=True)



def make_env(scenario_name, benchmark=False):
	scenario = scenarios.load(scenario_name + ".py").Scenario()
	world = scenario.make_world()
	if benchmark:
		env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation, scenario.benchmark_data, scenario.isFinished)
	else:
		env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation, None, scenario.isFinished)
	return env

'''
crossing_team_greedy
PRD_MAPPO_Q: value_lr = 5e-4; policy_lr = 5e-4; entropy_pen = 0.0; grad_clip_critic = 0.5; grad_clip_actor = 0.5; value_clip = 0.1; policy_clip = 0.1; ppo_epochs = 1; ppo_update_batch_size = 1

crossing_greedy
PRD_MAPPO_Q: value_lr = 5e-4; policy_lr = 5e-4; entropy_pen = 0.0; grad_clip_critic = 0.5; grad_clip_actor = 0.5; value_clip = 0.1; policy_clip = 0.1; ppo_epochs = 1; ppo_update_batch_size = 1
'''


if __name__ == '__main__':

	for i in range(1,6):
		extension = "MAPPO_Hard_Soft_Attn_run_"+str(i)
		test_num = "TEAM COLLISION AVOIDANCE"
		env_name = "crossing_team_greedy"
		experiment_type = "shared" # shared, prd_above_threshold, prd_top_k, prd_above_threshold_decay, prd_above_threshold_ascend

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
				"update_ppo_agent": 1, # update ppo agent after every update_ppo_agent episodes
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
				"max_episodes": 50000,
				"max_time_steps": 100,
				"experiment_type": experiment_type,
				"parallel_training": False,
				"scheduler_need": False,


				# ENVIRONMENT
				"team_size": 8,
				"env": env_name,

				# CRITIC
				"value_lr": 5e-4, #1e-3
				"grad_clip_critic": 10.0,
				"value_clip": 0.05,
				"enable_hard_attention": True,
				"num_heads": 4,
				"critic_weight_entropy_pen": 0.0,
				"lambda": 0.95, # 1 --> Monte Carlo; 0 --> TD(1)
				"norm_returns": False,
				

				# ACTOR
				"grad_clip_actor": 10.0,
				"policy_clip": 0.05,
				"policy_lr": 5e-4, #prd 1e-4
				"entropy_pen": 0.0, #8e-3
				"gae_lambda": 0.95,
				"select_above_threshold": 0.0,
				"threshold_min": 0.0, 
				"threshold_max": 0.0,
				"steps_to_take": 1000,
				"top_k": 0,
				"norm_adv": False,
			}

		seeds = [42, 142, 242, 342, 442]
		torch.manual_seed(seeds[dictionary["iteration"]-1])
		env = make_env(scenario_name=dictionary["env"],benchmark=False)
		ma_controller = MAPPO(env,dictionary)
		ma_controller.run()
