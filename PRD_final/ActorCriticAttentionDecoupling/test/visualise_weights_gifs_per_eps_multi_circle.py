import os
import numpy as np
import datetime
from typing import Any, List, Tuple, Union
import math

import torch
import torch.nn as nn
import torch.nn.functional as F 
import torch.optim as optim
from torch.distributions import Categorical
import torch.autograd as autograd
from torch.utils.tensorboard import SummaryWriter

from multiagent.environment import MultiAgentEnv
import multiagent.scenarios as scenarios

from a2c_test import *


def make_env(scenario_name, benchmark=False):
	# load scenario from script
	scenario = scenarios.load(scenario_name + ".py").Scenario()
	# scenario = Scenario()
	# create world
	world = scenario.make_world()
	# create multiagent environment
	if benchmark:
		env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation, scenario.benchmark_data, scenario.isFinished)
	else:
		env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation, None, scenario.isFinished)
	return env


def split_states(states, num_agents):
	states_critic = []
	states_actor = []
	for i in range(num_agents):
		states_critic.append(states[i][0])
		states_actor.append(states[i][1])

	states_critic = np.asarray(states_critic)
	states_actor = np.asarray(states_actor)

	return states_critic,states_actor


def make_gif(images,fname,fps=10, scale=1.0):
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




def calculate_indiv_weights(weights, num_agents, weight_dictionary):
	weights_per_agent = torch.sum(weights,dim=0) / weights.shape[0]

	for i in range(num_agents):
		agent_name = 'agent %d' % i
		for j in range(num_agents):
			agent_name_ = 'agent %d' % j
			weight_dictionary[agent_name][agent_name_] = weights_per_agent[i][j].item()


def run(env, max_steps):

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	num_agents = env.n
	num_actions = env.action_space[0].n


	obs_input_dim = 2*3
	obs_act_input_dim = obs_input_dim + num_actions
	obs_output_dim = obs_act_output_dim = 128
	final_input_dim = obs_act_output_dim
	final_output_dim = 1
	critic_network = MLPToGNNV6(obs_input_dim, obs_output_dim, obs_act_input_dim, obs_act_output_dim, final_input_dim, final_output_dim, num_agents, num_actions).to(device)

	# MLP POLICY
	policy_network = MLPPolicyNetwork(obs_input_dim, num_agents, num_actions).to(device)


	for exp_type in ["greedy_policy", "without_prd", "with_prd_soft_adv"]:
		for eps in ["5000", "10000", "25000", "50000", "100000", "125000", "150000", "175000", "200000"]:
			# Loading models
			model_path_value = "../../../../tests/multi_circle/models/multi_circular_"+exp_type+"_MLPToGNNV6_multi_circle_try2/critic_networks/05-07-2021VN_ATN_FCN_lr0.001_PN_ATN_FCN_lr0.0005_GradNorm0.5_Entropy0.008_trace_decay0.98topK_0select_above_threshold0.1softmax_cut_threshold0.1_epsiode"+eps+"_MLPToGNNV6.pt"
			model_path_policy = "../../../../tests/multi_circle/models/multi_circular_"+exp_type+"_MLPToGNNV6_multi_circle_try2/actor_networks/05-07-2021_PN_ATN_FCN_lr0.0005VN_SAT_FCN_lr0.001_GradNorm0.5_Entropy0.008_trace_decay0.98topK_0select_above_threshold0.1softmax_cut_threshold0.1_epsiode"+eps+"_MLPToGNNV6.pt"
			# For CPU
			# critic_network.load_state_dict(torch.load(model_path_value,map_location=torch.device('cpu')))
			# policy_network.load_state_dict(torch.load(model_path_policy,map_location=torch.device('cpu')))
			# For GPU
			critic_network.load_state_dict(torch.load(model_path_value))
			policy_network.load_state_dict(torch.load(model_path_policy))

			tensorboard_dir = '../../../../tests/vis_weights_gifs_per_eps_multi_cirlce_try2/runs/'+exp_type+'_'+eps+'/'
			writer = SummaryWriter(tensorboard_dir)


			gif_dir = '../../../../tests/vis_weights_gifs_per_eps_multi_cirlce_try2/gifs/'+exp_type+'_'+eps+'/'
			try: 
				os.makedirs(gif_dir, exist_ok = True) 
				print("Gif Directory created successfully") 
			except OSError as error: 
				print("Gif Directory can not be created")

			gif_path = gif_dir+'multi_circle.gif'

			weight_dictionary = {}

			for i in range(num_agents):
				agent_name = 'agent %d' % i
				weight_dictionary[agent_name] = {}
				for j in range(num_agents):
					agent_name_ = 'agent %d' % j
					weight_dictionary[agent_name][agent_name_] = 0


			states = env.reset()

			images = []

			states_critic,states_actor = split_states(states,num_agents)

			for step in range(max_steps):

				# At each step, append an image to list
				images.append(np.squeeze(env.render(mode='rgb_array')))
				
				actions = None
				dists = None
				with torch.no_grad():
					states_actor = torch.FloatTensor([states_actor]).to(device)
					dists, _ = policy_network.forward(states_actor)
					actions = [Categorical(dist).sample().cpu().detach().item() for dist in dists[0]]

					one_hot_actions = np.zeros((num_agents,num_actions))
					for i,act in enumerate(actions):
						one_hot_actions[i][act] = 1

					states_critic = torch.FloatTensor([states_critic]).to(device)
					V_values, weights = critic_network.forward(states_critic, dists.detach(), torch.FloatTensor(one_hot_actions).unsqueeze(0).to(device))

				# Advance a step and render a new image
				next_states,rewards,dones,info = env.step(actions)
				next_states_critic,next_states_actor = split_states(next_states, num_agents)

				total_rewards = np.sum(rewards)

				print("*"*100)
				print("TIMESTEP: {} | REWARD: {} \n".format(step,np.round(total_rewards,decimals=4)))
				print("*"*100)


				states_critic,states_actor = next_states_critic,next_states_actor
				states = next_states

				calculate_indiv_weights(weights, num_agents, weight_dictionary)
				for i in range(num_agents):
					agent_name = 'agent %d' % i
					writer.add_scalars('Weights_Critic/Average_Weights/'+agent_name,weight_dictionary[agent_name],step)

				
				# ENTROPY OF WEIGHTS
				entropy_weights = -torch.mean(torch.sum(weights * torch.log(torch.clamp(weights, 1e-10,1.0)), dim=2))
				writer.add_scalar('Weights_Critic/Entropy', entropy_weights.item(), step)

				writer.add_scalar('Reward Incurred/Reward',total_rewards,step)



				


			print("GENERATING GIF")
			make_gif(np.array(images),gif_path)


if __name__ == '__main__':
	env = make_env(scenario_name="multi_circular",benchmark=False)

	run(env,100)