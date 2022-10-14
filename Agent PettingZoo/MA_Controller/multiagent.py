import os
import time
from comet_ml import Experiment
import numpy as np
from agent import Agent
import torch
import datetime
from torch.distributions import Categorical



class MultiAgent:

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
		self.num_agents = len(self.env.agents)
		self.num_actions = 21
		self.date_time = f"{datetime.datetime.now():%d-%m-%Y}"
		self.env_name = dictionary["env"]
		self.test_num = dictionary["test_num"]
		self.max_episodes = dictionary["max_episodes"]
		self.max_time_steps = dictionary["max_time_steps"]
		self.experiment_type = dictionary["experiment_type"]
		self.update_ppo_agent = dictionary["update_ppo_agent"]
		self.model_path_value = dictionary["model_path_value"]
		self.model_path_policy = dictionary["model_path_policy"]
		self.update_type = dictionary["update_type"]

		self.agent_names = []
		for i in range(self.num_agents//2):
			self.agent_names.append("red_"+str(i))
			self.agent_names.append("blue_"+str(i))

		self.no_op_action_num = 0
		self.obs_dim = 6929


		self.comet_ml = None
		if self.save_comet_ml_plot:
			self.comet_ml = Experiment("im5zK8gFkz6j07uflhc3hXk8I",project_name=dictionary["test_num"])
			self.comet_ml.log_parameters(dictionary)


		self.agents = Agent(self.env, dictionary, self.comet_ml)

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


	def change_observation(self, observation):
		observation = observation.tolist()
		new_list = []
		for i in range(len(observation)):
			for j in range(len(observation[i])):
				for k in range(len(observation[i][j])):
					new_list.append(observation[i][j][k])
		new_observation = np.array(new_list).astype(np.float)
		return new_observation




	def run(self):  
		if self.eval_policy:
			self.rewards = {0:[], 1:[]}
			self.rewards_mean_per_1000_eps = {0:[], 1:[]}
			self.num_agents_alive = {0:[], 1:[]}
			self.num_agents_alive_mean_per_1000_eps = {0:[], 1:[]}

		for episode in range(1,self.max_episodes+1):

			observations = self.env.reset()

			trajectory = []
			episode_reward = [0.0, 0.0]
			final_timestep = self.max_time_steps
			actions = {}
			for step in range(1, self.max_time_steps+1):

				observations_ = []
				actions_ = []
				probs_ = []
				logprobs_ = []
				one_hot_actions = np.zeros((self.num_agents,self.num_actions))
				# for agent_num, agent in enumerate(self.env.agents):
				# 	agent_observation = self.change_observation(observations[agent])
				# 	observations_.append(agent_observation)

				# 	action, probs, action_logprob = self.agents.get_action(agent_observation, greedy=False)
				# 	actions[agent] = action

				# 	actions_.append(action)
				# 	probs_.append(probs)
				# 	logprobs_.append(action_logprob)
				# 	one_hot_actions[agent_num][action] = 1

				# rewards_ = list(rewards.values())
				# dones_ = list(dones.values())

				for agent_num, agent in enumerate(self.agent_names):
					if agent in self.env.agents:
						agent_observation = self.change_observation(observations[agent])
						action, probs, action_logprob = self.agents.get_action(agent_observation, greedy=False)
					else:
						agent_observation = np.zeros(self.obs_dim)
						action = 0
						dists = torch.zeros(self.num_actions)
						# set no-op action with max prob
						dists[0] = 1
						probs = Categorical(dists)
						action_logprob = probs.log_prob(torch.FloatTensor([action])).numpy()
						probs = dists.numpy()

					observations_.append(agent_observation)

					actions[agent] = action

					actions_.append(action)
					probs_.append(probs)
					logprobs_.append(action_logprob)
					one_hot_actions[agent_num][action] = 1

				new_observations, rewards, dones, infos = self.env.step(actions)

				rewards_ = []
				dones_ = []
				for agent_num, agent in enumerate(self.agent_names):
					if agent in self.env.agents:
						rewards_.append(rewards[agent])
						dones_.append(dones[agent])
					else:
						rewards_.append(0)
						dones_.append(True)
				

				# record data
				self.agents.buffer.observations.append(observations_)
				self.agents.buffer.actions.append(actions_)
				self.agents.buffer.one_hot_actions.append(one_hot_actions)
				self.agents.buffer.probs.append(probs_)
				self.agents.buffer.logprobs.append(logprobs_)
				self.agents.buffer.rewards.append(rewards_)
				self.agents.buffer.dones.append(dones_)


				if not self.env.agents:  
					break

				for agent in self.env.agents: 
					if "red" in agent: 
						team = 0
					else: 
						team = 1
					
					episode_reward[team] = episode_reward[team] + rewards[agent]


				observations = new_observations

			
				if all(dones_) or step == self.max_time_steps:

					team_size = self.env.team_size()

					print("*"*100)
					print("EPISODE: {} | REWARD TEAM 1: {} | REWARD TEAM 2: {} | TIME TAKEN: {} / {} \n".format(episode,np.round(episode_reward[0],decimals=4),np.round(episode_reward[0],decimals=4),step,self.max_time_steps))
					print("NUM AGENTS ALIVE TEAM 1: {} | NUM AGENTS ALIVE TEAM 2: {} \n".format(team_size[0], team_size[1]))
					print("*"*100)

					final_timestep = step

					if self.save_comet_ml_plot:
						self.comet_ml.log_metric('Episode_Length', step, episode)
						self.comet_ml.log_metric('Reward Team 1', episode_reward[0], episode)
						self.comet_ml.log_metric('Reward Team 2', episode_reward[1], episode)

					break

			if self.eval_policy:
				self.rewards[0].append(episode_reward[0])
				self.rewards[1].append(episode_reward[1])
				self.num_agents_alive[0].append(team_size[0])
				self.num_agents_alive[0].append(team_size[1])

			if episode > self.save_model_checkpoint and self.eval_policy:
				self.rewards_mean_per_1000_eps[0].append(sum(self.rewards[0][episode-self.save_model_checkpoint:episode])/self.save_model_checkpoint)
				self.rewards_mean_per_1000_eps[1].append(sum(self.rewards[1][episode-self.save_model_checkpoint:episode])/self.save_model_checkpoint)
				self.num_agents_alive_mean_per_1000_eps[0].append(sum(self.num_agents_alive[0][episode-self.save_model_checkpoint:episode])/self.save_model_checkpoint)
				self.num_agents_alive_mean_per_1000_eps[1].append(sum(self.num_agents_alive[1][episode-self.save_model_checkpoint:episode])/self.save_model_checkpoint)
	

			if not(episode%self.save_model_checkpoint) and episode!=0 and self.save_model:	
				torch.save(self.agents.critic_network.state_dict(), self.critic_model_path+'_epsiode'+str(episode)+'.pt')
				torch.save(self.agents.policy_network.state_dict(), self.actor_model_path+'_epsiode'+str(episode)+'.pt')  

			if self.learn and not(episode%self.update_ppo_agent) and episode != 0:
				if self.update_type == "ppo":
					self.agents.update(episode) 
				elif self.update_type == "a2c":
					self.agents.a2c_update(episode) 


			if self.eval_policy and not(episode%self.save_model_checkpoint) and episode!=0:
				np.save(os.path.join(self.policy_eval_dir,self.test_num+"reward_list"), np.array(self.rewards), allow_pickle=True, fix_imports=True)
				np.save(os.path.join(self.policy_eval_dir,self.test_num+"mean_rewards_per_1000_eps"), np.array(self.rewards_mean_per_1000_eps), allow_pickle=True, fix_imports=True)