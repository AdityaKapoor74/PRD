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


def split_states(states,num_agents):
	states_agent = []
	states_goal = []
	for i in range(num_agents):
		states_agent.append(states[i][0])
		states_goal.append(states[i][1])

	states_agent = np.asarray(states_agent)
	states_goal = np.asarray(states_goal)

	return states_agent,states_goal


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




def calculate_indiv_weights_agent_agent(weights, num_agents, weight_agent_agent_dictionary):
	weights_per_agent = torch.sum(weights,dim=0) / weights.shape[0]

	for i in range(num_agents):
		agent_name = 'agent %d' % i
		for j in range(num_agents):
			agent_name_ = 'agent %d' % j
			weight_agent_agent_dictionary[agent_name][agent_name_] = weights_per_agent[i][j].item()

def calculate_indiv_weights_agent_goal(weights, num_agents, weight_agent_goal_dictionary):
	weights_per_agent = torch.sum(weights,dim=0) / weights.shape[0]

	for i in range(num_agents):
		agent_name = 'agent %d' % i
		for j in range(num_agents):
			goal_name = 'goal %d' % j
			weight_agent_goal_dictionary[agent_name][goal_name] = weights_per_agent[i][j].item()


def run(env, max_steps):

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	num_agents = env.n
	num_actions = env.action_space[0].n


	obs_agent_input_dim = 2*2+1
	obs_goal_input_dim = 2+1
	obs_act_input_dim = obs_agent_input_dim + num_actions
	obs_agent_output_dim = obs_goal_output_dim = obs_act_output_dim = 128
	final_input_dim = obs_goal_output_dim + obs_act_output_dim
	final_output_dim = 1
	critic_network = GATSocialDilemma(obs_agent_input_dim, obs_agent_output_dim, obs_goal_input_dim, obs_goal_output_dim, obs_act_input_dim, obs_act_output_dim, final_input_dim, final_output_dim, num_agents, num_agents, num_actions, threshold=0.1).to(device)

	# MLP POLICY
	policy_network = MLPPolicyNetworkSocialDilemma(2*2+1, num_agents, 2+1, num_agents, num_actions).to(device)

	# Loading models
	# model_path_value = "../../../../tests/color_social_dilemma_DualGAT_try2/models/color_social_dilemma_with_prd_soft_adv_GATSocialDilemma_4Agents_4Teams/critic_networks/03-07-2021VN_ATN_FCN_lr0.001_PN_ATN_FCN_lr0.0005_GradNorm0.5_Entropy0.008_trace_decay0.98topK_0select_above_threshold0.1softmax_cut_threshold0.1_epsiode5000_GATSocialDilemma.pt"
	# model_path_policy = "../../../../tests/color_social_dilemma_DualGAT_try2/models/color_social_dilemma_with_prd_soft_adv_GATSocialDilemma_4Agents_4Teams/actor_networks/03-07-2021_PN_ATN_FCN_lr0.0005VN_SAT_FCN_lr0.001_GradNorm0.5_Entropy0.008_trace_decay0.98topK_0select_above_threshold0.1softmax_cut_threshold0.1_epsiode5000_GATSocialDilemma.pt"
	# For CPU
	# critic_network.load_state_dict(torch.load(model_path_value,map_location=torch.device('cpu')))
	# policy_network.load_state_dict(torch.load(model_path_policy,map_location=torch.device('cpu')))
	# For GPU
	# critic_network.load_state_dict(torch.load(model_path_value))
	# policy_network.load_state_dict(torch.load(model_path_policy))

	tensorboard_dir = '../../../../tests/vis_weights_gifs_per_eps_social_dilemma_try2/runs/with_prd_soft_adv_5000/'
	writer = SummaryWriter(tensorboard_dir)


	gif_dir = '../../../../tests/vis_weights_gifs_per_eps_social_dilemma_try2/gifs/with_prd_soft_adv_5000/'
	try: 
		os.makedirs(gif_dir, exist_ok = True) 
		print("Gif Directory created successfully") 
	except OSError as error: 
		print("Gif Directory can not be created")

	gif_path = gif_dir+'color_social_dilemma.gif'

	weight_agent_agent_dictionary = {}
	for i in range(num_agents):
		agent_name = 'agent %d' % i
		weight_agent_agent_dictionary[agent_name] = {}
		for j in range(num_agents):
			agent_name_ = 'agent %d' % j
			weight_agent_agent_dictionary[agent_name][agent_name_] = 0

	weight_agent_goal_dictionary = {}
	for i in range(num_agents):
		agent_name = 'agent %d' % i
		weight_agent_goal_dictionary[agent_name] = {}
		for j in range(num_agents):
			goal_name = 'goal %d' % j
			weight_agent_goal_dictionary[agent_name][goal_name] = 0


	states = env.reset()

	images = []

	states_agent,states_goal = split_states(states,num_agents)
	states_goal = states_goal[0]

	for step in range(max_steps):

		# At each step, append an image to list
		images.append(np.squeeze(env.render(mode='rgb_array')))
		
		actions = None
		dists = None
		with torch.no_grad():
			state_agent = torch.FloatTensor([states_agent]).to(device)
			state_goal = torch.FloatTensor([states_goal]).to(device)
			dists = policy_network.forward(state_agent, state_goal)
			actions = [Categorical(dist).sample().cpu().detach().item() for dist in dists[0]]

			one_hot_actions = np.zeros((num_agents,num_actions))
			for i,act in enumerate(actions):
				one_hot_actions[i][act] = 1

			V_values, weights_agent_goal, weights_agent_agent = critic_network.forward(state_agent, state_goal, dists.detach(), torch.FloatTensor(one_hot_actions).unsqueeze(0).to(device))
			# V_values, weights = critic_network.forward(torch.FloatTensor(states_critic).unsqueeze(0).to(device), dists.detach(), torch.FloatTensor(one_hot_actions).unsqueeze(0).to(device))


		# Advance a step and render a new image
		next_states,rewards,dones,info = env.step(actions)
		next_states_agent,next_states_goal = split_states(next_states, num_agents)
		next_states_goal = next_states_goal[0]

		total_rewards = np.sum(rewards)

		print("*"*100)
		print("TIMESTEP: {} | REWARD: {} \n".format(step,np.round(total_rewards,decimals=4)))
		print("*"*100)


		states_agent,states_goal = next_states_agent,next_states_goal
		states = next_states

		calculate_indiv_weights_agent_agent(weights_agent_agent, num_agents, weight_agent_agent_dictionary)
		for i in range(num_agents):
			agent_name = 'agent %d' % i
			writer.add_scalars('Weights_Critic/Average_Weights_Agent_Agent/'+agent_name,weight_agent_agent_dictionary[agent_name],step)

		calculate_indiv_weights_agent_goal(weights_agent_goal, num_agents, weight_agent_goal_dictionary)
		for i in range(num_agents):
			agent_name = 'agent %d' % i
			writer.add_scalars('Weights_Critic/Average_Weights_Agent_Goal/'+agent_name,weight_agent_goal_dictionary[agent_name],step)
		
		# ENTROPY OF WEIGHTS
		entropy_weights = -torch.mean(torch.sum(weights_agent_goal * torch.log(torch.clamp(weights_agent_goal, 1e-10,1.0)), dim=2))
		writer.add_scalar('Weights_Critic/Entropy_Agent_Goal', entropy_weights.item(), step)

		entropy_weights = -torch.mean(torch.sum(weights_agent_agent * torch.log(torch.clamp(weights_agent_agent, 1e-10,1.0)), dim=2))
		writer.add_scalar('Weights_Critic/Entropy_Agent_Agent', entropy_weights.item(), step)

		writer.add_scalar('Reward Incurred/Reward',total_rewards,step)



		


	print("GENERATING GIF")
	make_gif(np.array(images),gif_path)


if __name__ == '__main__':
	env = make_env(scenario_name="color_social_dilemma",benchmark=False)

	run(env,500)