import os
import torch
import torch.nn.functional as F 
import torch.optim as optim
from torch.distributions import Categorical
import torch.autograd as autograd
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import datetime


from multiagent.environment import MultiAgentEnv
# from multiagent.scenarios.simple_spread import Scenario
import multiagent.scenarios as scenarios
import torch 
import numpy as np


# MODELS
from typing import Any, List, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import datetime
import math

from os import listdir
from os.path import isfile, join
import pickle, simplejson
from a2c_collision_avoidance import ScalarDotProductPolicyNetwork


eval_after = [i*1000 for i in range(1,80)]


def make_env(scenario_name, benchmark=False):
	# load scenario from script
	scenario = scenarios.load(scenario_name + ".py").Scenario()
	# scenario = Scenario()
	# create world
	world = scenario.make_world()
	# create multiagent environment
	if benchmark:
		env = MultiAgentEnv(world, scenario.reset_world, scenario.reward_paired_agents, scenario.observation, scenario.benchmark_data, scenario.isFinished)
	else:
		env = MultiAgentEnv(world, scenario.reset_world, scenario.reward_paired_agents, scenario.observation, None, scenario.isFinished)
	return env


def split_states(states, num_agents):

	states_critic = []
	states_actor = []
	for i in range(num_agents):
		states_critic.append(states[i][0])
		states_actor.append(states[i][1])

	states_critic = np.asarray(states_critic)
	states_actor = np.asarray(states_actor)

	return states_critic, states_actor



def run(env, max_episodes, max_steps):  


	# cut the tail of softmax --> Used in softmax with normalization
	softmax_cut_threshold = 0.1

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	num_agents = env.n
	num_actions = env.action_space[0].n

	obs_input_dim = 2*3
	obs_output_dim = 16
	final_input_dim = obs_output_dim
	final_output_dim = num_actions
	policy_network = ScalarDotProductPolicyNetwork(obs_input_dim, obs_output_dim, final_input_dim, final_output_dim, num_agents, num_actions, softmax_cut_threshold).to(device)

	policy_eval_dir = '../../../policy_eval/paired_by_sharing_goals/'
	try: 
		os.makedirs(policy_eval_dir, exist_ok = True) 
		print("Policy Eval Directory created successfully") 
	except OSError as error: 
		print("Policy Eval Directory can not be created")


	experiment_type = ["without_prd", "with_prd_top1", "with_prd_top3", "with_prd_soft_adv", "without_prd_scaled", "with_prd_top1_scaled", "with_prd_top3_scaled", "with_prd_soft_adv_scaled"]

	runs = ["v1", "v2", "v3", "v4", "v5"]

	for run in runs:
		for experiment in experiment_type:

			# Loading models
			# FOR LOCAL SYSTEM
			# model_dir_policy = "../../../remote_stations/collision_avoidance/"+run+"/models/Scalar_dot_product/collision_avoidance/6_Agents/SingleAttentionMechanism/" + experiment + "/actor_networks/"
			# FOR REMOTE SYSTEM
			model_dir_policy = "../../../all_models/models/models_"+run+"/" + experiment + "/actor_networks/"
			
			policy_eval_file_path = policy_eval_dir+'paired_by_sharing_goals_10_Agents' + experiment + '.txt'

			onlyfiles = [f for f in listdir(model_dir_policy) if isfile(join(model_dir_policy, f))]

			rewards_dict = {}
			time_steps_dict = {}

			for file in onlyfiles:

				episode_num = file.split('_')[-1][7:-3]
				model_path_policy = model_dir_policy + file
				# For CPU
				# policy_network.load_state_dict(torch.load(model_path_policy,map_location=torch.device('cpu')))
				# For GPU
				policy_network.load_state_dict(torch.load(model_path_policy))

				rewards_list = []
				time_steps_list = []

				for episode in range(1,max_episodes+1):

					states = env.reset()

					states_critic,states_actor = split_states(states, num_agents)

					total_rewards = 0
					final_time_step = max_steps

					for step in range(1,max_steps+1):
						
						actions = None
						dists = None
						with torch.no_grad():
							states_actor = torch.FloatTensor([states_actor]).to(device)
							dists, _ = policy_network.forward(states_actor)
							actions = [Categorical(dist).sample().cpu().detach().item() for dist in dists[0]]

						next_states,rewards,dones,info = env.step(actions)
						next_states_critic,next_states_actor = split_states(next_states, num_agents)

						total_rewards += np.sum(rewards)


						if all(dones):
							final_time_step = step
							break

						states_critic,states_actor = next_states_critic,next_states_actor
						states = next_states

					print("*"*50)
					print("RUN NUMBER", run, "EXPERIMENT", experiment, "MODEL EPISODE", episode_num, "EPISODE REWARD", total_rewards, "FINAL TIMESTEP", final_time_step)
					print("*"*50)

					rewards_list.append(total_rewards)
					time_steps_list.append(final_time_step)

				rewards_dict[episode_num] = sum(rewards_list)/len(rewards_list)
				time_steps_dict[episode_num] = sum(time_steps_list)/len(time_steps_list)

			rewards = []
			time_steps = []
			for ep in eval_after:
				rewards.append(rewards_dict[str(ep)])
				time_steps.append(time_steps_dict[str(ep)])

			with open(policy_eval_file_path, "a+") as fp:
				fp.write("REWARDS \n")
				simplejson.dump(rewards, fp)
				fp.write("\n")
				fp.write("TIME STEPS \n")
				simplejson.dump(time_steps, fp)
				fp.write("\n")

			print("SAVED DATA!")

if __name__ == '__main__':
	env = make_env(scenario_name="paired_by_sharing_goals",benchmark=False)

	run(env, max_episodes=1000, max_steps=100)