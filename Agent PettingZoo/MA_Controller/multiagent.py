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
			self.rewards = {"red":[], "blue":[]}
			self.rewards_mean_per_1000_eps = {"red":[], "blue":[]}
			self.num_agents_alive = {"red":[], "blue":[]}
			self.num_agents_alive_mean_per_1000_eps = {"red":[], "blue":[]}

		for episode in range(1,self.max_episodes+1):

			observations = self.env.reset()

			trajectory = []
			episode_reward = [0.0, 0.0]
			final_timestep = self.max_time_steps
			actions = {}
			for step in range(1, self.max_time_steps+1):

				observations_red = []
				observations_blue = []
				actions_red = []
				actions_blue = []
				probs_red = []
				probs_blue = []
				logprobs_red = []
				logprobs_blue = []
				one_hot_actions_red = np.zeros((self.num_agents//2,self.num_actions))
				one_hot_actions_blue = np.zeros((self.num_agents//2,self.num_actions))
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
						action, probs, action_logprob = self.agents.get_action(agent_observation, agent, greedy=False)
					else:
						agent_observation = np.zeros(self.obs_dim)
						action = 0
						dists = torch.zeros(self.num_actions)
						# set no-op action with max prob
						dists[0] = 1
						probs = Categorical(dists)
						action_logprob = probs.log_prob(torch.FloatTensor([action])).numpy()
						probs = dists.numpy()

					actions[agent] = action

					if "red" in agent:
						observations_red.append(agent_observation)
						actions_red.append(action)
						probs_red.append(probs)
						logprobs_red.append(action_logprob)
						one_hot_actions_red[int(agent[4:])][action] = 1
					elif "blue" in agent:
						observations_blue.append(agent_observation)
						actions_blue.append(action)
						probs_blue.append(probs)
						logprobs_blue.append(action_logprob)
						one_hot_actions_blue[int(agent[5:])][action] = 1

				new_observations, rewards, dones, infos = self.env.step(actions)

				rewards_red = []
				rewards_blue = []
				dones_red = []
				dones_blue = []
				for agent_num, agent in enumerate(self.agent_names):
					if agent in self.env.agents:
						reward = rewards[agent]
						done = dones[agent]
					else:
						reward = 0.0
						done = True

					if "red" in agent:
						rewards_red.append(reward)
						dones_red.append(done)
					elif "blue" in agent:
						rewards_blue.append(reward)
						dones_blue.append(done)
				

				# record data
				self.agents.buffer.observations_red.append(observations_red)
				self.agents.buffer.actions_red.append(actions_red)
				self.agents.buffer.one_hot_actions_red.append(one_hot_actions_red)
				self.agents.buffer.probs_red.append(probs_red)
				self.agents.buffer.logprobs_red.append(logprobs_red)
				self.agents.buffer.rewards_red.append(rewards_red)
				self.agents.buffer.dones_red.append(dones_red)
				
				self.agents.buffer.observations_blue.append(observations_blue)
				self.agents.buffer.actions_blue.append(actions_blue)
				self.agents.buffer.one_hot_actions_blue.append(one_hot_actions_blue)
				self.agents.buffer.probs_blue.append(probs_blue)
				self.agents.buffer.logprobs_blue.append(logprobs_blue)
				self.agents.buffer.rewards_blue.append(rewards_blue)
				self.agents.buffer.dones_blue.append(dones_blue)


				if not self.env.agents:  
					break

				for agent in self.env.agents: 
					if "red" in agent: 
						team = 0
					else: 
						team = 1
					
					episode_reward[team] = episode_reward[team] + rewards[agent]


				observations = new_observations

			
				if all(dones_red) or all(dones_blue) or step == self.max_time_steps:

					team_size = self.env.team_size()

					print("*"*100)
					print("EPISODE: {} | REWARD TEAM RED: {} | REWARD TEAM BLUE: {} | TIME TAKEN: {} / {} \n".format(episode,np.round(episode_reward[0],decimals=4),np.round(episode_reward[1],decimals=4),step,self.max_time_steps))
					print("NUM AGENTS ALIVE TEAM RED: {} | NUM AGENTS ALIVE TEAM BLUE: {} \n".format(team_size[0], team_size[1]))
					print("*"*100)

					final_timestep = step

					if self.save_comet_ml_plot:
						self.comet_ml.log_metric('Episode_Length', step, episode)
						self.comet_ml.log_metric('Reward Team Red', episode_reward[0], episode)
						self.comet_ml.log_metric('Reward Team Blue', episode_reward[1], episode)
						self.comet_ml.log_metric('Team Red Size', team_size[0], episode)
						self.comet_ml.log_metric('Team Blue Size', team_size[1], episode)

					break

			if self.eval_policy:
				self.rewards["red"].append(episode_reward[0])
				self.rewards["blue"].append(episode_reward[1])
				self.num_agents_alive["red"].append(team_size[0])
				self.num_agents_alive["blue"].append(team_size[1])

			if episode > self.save_model_checkpoint and self.eval_policy:
				self.rewards_mean_per_1000_eps["red"].append(sum(self.rewards["red"][episode-self.save_model_checkpoint:episode])/self.save_model_checkpoint)
				self.rewards_mean_per_1000_eps["blue"].append(sum(self.rewards["blue"][episode-self.save_model_checkpoint:episode])/self.save_model_checkpoint)
				self.num_agents_alive_mean_per_1000_eps["red"].append(sum(self.num_agents_alive["red"][episode-self.save_model_checkpoint:episode])/self.save_model_checkpoint)
				self.num_agents_alive_mean_per_1000_eps["blue"].append(sum(self.num_agents_alive["blue"][episode-self.save_model_checkpoint:episode])/self.save_model_checkpoint)
	

			if not(episode%self.save_model_checkpoint) and episode!=0 and self.save_model:	
				torch.save(self.agents.critic_network_red.state_dict(), self.critic_model_path+'_red_epsiode'+str(episode)+'.pt')
				torch.save(self.agents.policy_network_red.state_dict(), self.actor_model_path+'_red_epsiode'+str(episode)+'.pt')

				torch.save(self.agents.critic_network_blue.state_dict(), self.critic_model_path+'_blue_epsiode'+str(episode)+'.pt')
				torch.save(self.agents.policy_network_blue.state_dict(), self.actor_model_path+'_blue_epsiode'+str(episode)+'.pt')  

			if self.learn and not(episode%self.update_ppo_agent) and episode != 0:
				if self.update_type == "ppo":
					self.agents.update(episode) 
				elif self.update_type == "a2c":
					self.agents.a2c_update(episode) 


			if self.eval_policy and not(episode%self.save_model_checkpoint) and episode!=0:
				np.save(os.path.join(self.policy_eval_dir,self.test_num+"red_reward"), np.array(self.rewards["red"]), allow_pickle=True, fix_imports=True)
				np.save(os.path.join(self.policy_eval_dir,self.test_num+"red_mean_rewards_per_1000_eps"), np.array(self.rewards_mean_per_1000_eps["red"]), allow_pickle=True, fix_imports=True)
				np.save(os.path.join(self.policy_eval_dir,self.test_num+"blue_reward"), np.array(self.rewards["blue"]), allow_pickle=True, fix_imports=True)
				np.save(os.path.join(self.policy_eval_dir,self.test_num+"blue_mean_rewards_per_1000_eps"), np.array(self.rewards_mean_per_1000_eps["blue"]), allow_pickle=True, fix_imports=True)

				np.save(os.path.join(self.policy_eval_dir,self.test_num+"red_num_agents_alive"), np.array(self.num_agents_alive["red"]), allow_pickle=True, fix_imports=True)
				np.save(os.path.join(self.policy_eval_dir,self.test_num+"red_num_agents_alive_per_1000_eps"), np.array(self.num_agents_alive_mean_per_1000_eps["red"]), allow_pickle=True, fix_imports=True)
				np.save(os.path.join(self.policy_eval_dir,self.test_num+"blue_num_agents_alive"), np.array(self.num_agents_alive["blue"]), allow_pickle=True, fix_imports=True)
				np.save(os.path.join(self.policy_eval_dir,self.test_num+"blue_mean_rewards_per_1000_eps"), np.array(self.num_agents_alive_mean_per_1000_eps["blue"]), allow_pickle=True, fix_imports=True)