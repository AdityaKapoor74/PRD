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

from a2c_crowd_nav import ScalarDotProductPolicyNetwork, ScalarDotProductCriticNetwork


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


def split_states(states, num_entities, for_who):

		states_critic = []
		states_actor = []

		if for_who == "agents":
			for i in range(num_entities):
				states_critic.append(states[i][0])
				states_actor.append(states[i][1])
		elif for_who == "people":
			for i in range(num_entities):
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




def calculate_indiv_weights(weights, weight_dictionary_agent, num_agents, weight_dictionary_people, num_people):

	if num_agents is not None and weight_dictionary_agent is not None:
		for i in range(num_agents):
			agent_name = 'agent %d' % i
			for j in range(num_agents):
				agent_name_ = 'agent %d' % j
				weight_dictionary_agent[agent_name][agent_name_] = weights[i][j].item()

	if num_people is not None and weight_dictionary_people is not None:
		for i in range(num_agents):
			agent_name = 'agent %d' % i
			for j in range(num_people):
				person_name = 'people %d' % j
				weight_dictionary_people[agent_name][person_name] = weights[i][j].item()



def run(env, max_steps):  


	# cut the tail of softmax --> Used in softmax with normalization
	softmax_cut_threshold = 0.1

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	num_agents = 6
	num_people = 4
	num_actions = env.action_space[0].n

	# SCALAR DOT PRODUCT
	obs_agent_input_dim = 2*3
	obs_act_agent_input_dim = obs_agent_input_dim + num_actions # (pose,vel,goal pose, paired agent goal pose) --> observations 
	obs_act_agent_output_dim = 16
	obs_people_input_dim = 2*2
	obs_act_people_input_dim = obs_people_input_dim + num_actions # (pose,vel,goal pose, paired agent goal pose) --> observations 
	obs_act_people_output_dim = 16
	final_input_dim = obs_act_agent_output_dim + obs_act_people_output_dim
	final_output_dim = 1
	critic_network = ScalarDotProductCriticNetwork(obs_act_agent_input_dim, obs_act_agent_output_dim, obs_act_people_input_dim, obs_act_people_output_dim, final_input_dim, final_output_dim, num_agents, num_people, num_actions, softmax_cut_threshold).to(device)
	
	
	# SCALAR DOT PRODUCT POLICY NETWORK
	obs_agent_input_dim = 2*3
	obs_agent_output_dim = 16
	obs_people_input_dim = 2*2 # (pose,vel,goal pose, paired agent goal pose) --> observations 
	obs_people_output_dim = 16
	final_input_dim = obs_agent_output_dim + obs_people_output_dim
	final_output_dim = num_actions
	policy_network = ScalarDotProductPolicyNetwork(obs_agent_input_dim, obs_agent_output_dim, obs_people_input_dim, obs_people_output_dim, final_input_dim, final_output_dim, num_agents, num_people, num_actions, softmax_cut_threshold).to(device)



	# Loading models
	# model_path_value = "../../../models/Scalar_dot_product/crowd_nav/6_Agents_4_People/SingleAttentionMechanism/with_prd_soft_adv/critic_networks/24-05-2021VN_ATN_FCN_lr0.01_PN_ATN_FCN_lr0.001_GradNorm0.5_Entropy0.008_trace_decay0.98topK_2select_above_threshold0.1softmax_cut_threshold0.1_epsiode80000.pt"
	# model_path_policy = "../../../models/Scalar_dot_product/crowd_nav/6_Agents_4_People/SingleAttentionMechanism/with_prd_soft_adv/actor_networks/24-05-2021_PN_ATN_FCN_lr0.001VN_SAT_FCN_lr0.01_GradNorm0.5_Entropy0.008_trace_decay0.98topK_2select_above_threshold0.1softmax_cut_threshold0.1_epsiode80000.pt"
	# For CPU
	# critic_network.load_state_dict(torch.load(model_path_value,map_location=torch.device('cpu')))
	# policy_network.load_state_dict(torch.load(model_path_policy,map_location=torch.device('cpu')))
	# # For GPU
	# critic_network.load_state_dict(torch.load(model_path_value))
	# policy_network.load_state_dict(torch.load(model_path_policy))

	tensorboard_dir = '../../../runs/plot_weights_gif_crowd_nav/Scalar_dot_product/crowd_nav/6_Agents_4_People/SingleAttentionMechanism/without_prd/'
	writer = SummaryWriter(tensorboard_dir)


	gif_dir = '../../../gifs/plot_weights_gif_crowd_nav/Scalar_dot_product/crowd_nav/6_Agents_4_People/SingleAttentionMechanism/without_prd/'
	try: 
		os.makedirs(gif_dir, exist_ok = True) 
		print("Gif Directory created successfully") 
	except OSError as error: 
		print("Gif Directory can not be created")

	gif_path = gif_dir+'crowd_nav_without_prd.gif'

	weight_dictionary_agent = {}

	for i in range(num_agents):
		agent_name = 'agent %d' % i
		weight_dictionary_agent[agent_name] = {}
		for j in range(num_agents):
			agent_name_ = 'agent %d' % j
			weight_dictionary_agent[agent_name][agent_name_] = 0

	weight_dictionary_people = {}

	for i in range(num_agents):
		agent_name = 'agent %d' % i
		weight_dictionary_people[agent_name] = {}
		for j in range(num_people):
			people_name = 'people %d' % j
			weight_dictionary_people[agent_name][people_name] = 0


	states = env.reset()

	states_agents = states[:num_agents]
	states_people = states[num_agents:]

	images = []

	total_rewards = 0

	states_critic,states_actor = split_states(states_agents, num_agents, "agents")
	states_critic_people,states_actor_people = split_states(states_people, num_people, "people")

	for step in range(max_steps):

		# At each step, append an image to list
		images.append(np.squeeze(env.render(mode='rgb_array')))
		
		actions = []
		dists = None
		with torch.no_grad():
			states_actor = torch.FloatTensor([states_actor]).to(device)
			states_actor_people = torch.FloatTensor([states_actor_people]).to(device)
			dists, _, _ = policy_network.forward(states_actor, states_actor_people)
			actions = [Categorical(dist).sample().cpu().detach().item() for dist in dists[0]]

		one_hot_actions = np.zeros((num_agents,num_actions))
		for i,act in enumerate(actions):
			one_hot_actions[i][act] = 1

		actions_people = []
		for i in range(num_people):
			if i%4 == 0:
				actions_people.append(2)
			elif i%4 == 1:
				actions_people.append(4)
			elif i%4 == 2:
				actions_people.append(1)
			elif i%4 == 3:
				actions_people.append(3)
		one_hot_actions_people = np.zeros((num_people,num_actions))
		for i,act in enumerate(actions_people):
			one_hot_actions_people[i][act] = 1


		next_states,rewards,dones,info = env.step(actions+actions_people)
		next_states_agents = next_states[:num_agents]
		next_states_people = next_states[num_agents:]
		next_states_critic,next_states_actor = split_states(next_states_agents, num_agents, "agents")
		next_states_critic_people,next_states_actor_people = split_states(next_states_people, num_people, "people")

		rewards = rewards[:num_agents]
		dones = dones[:num_agents]
		total_rewards += np.sum(rewards)
			

		with torch.no_grad():
			V_values, weights_agents, weights_people = critic_network.forward(torch.FloatTensor(states_critic).unsqueeze(0).to(device), dists.detach(), torch.FloatTensor(one_hot_actions).unsqueeze(0).to(device), torch.FloatTensor(states_critic_people).unsqueeze(0).to(device), torch.FloatTensor(one_hot_actions_people).unsqueeze(0).to(device))

		print("*"*100)
		print("TIMESTEP: {} | REWARD: {} \n".format(step,np.round(total_rewards,decimals=4)))
		print("*"*100)


		states_critic,states_actor = next_states_critic,next_states_actor
		states_critic_people,states_actor_people = next_states_critic_people,next_states_actor_people
		states = next_states

		writer.add_scalar('Reward Incurred/Reward',total_rewards,step)

		calculate_indiv_weights(weights_agents.squeeze(0), weight_dictionary_agent,num_agents, None, None)
		calculate_indiv_weights(weights_people.squeeze(0).squeeze(-1), None, num_agents, weight_dictionary_people,num_people)

		for i in range(num_agents):
			agent = 'agent %d' % i
			writer.add_scalars('Weights/Average_Weights_Agents/'+agent,weight_dictionary_agent[agent],step)
			writer.add_scalars('Weights/Average_Weights_People/'+agent,weight_dictionary_people[agent],step)


		


	print("GENERATING GIF")
	make_gif(np.array(images),gif_path)


if __name__ == '__main__':
	env = make_env(scenario_name="crowd_nav",benchmark=False)

	run(env,100)