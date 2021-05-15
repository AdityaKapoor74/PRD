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
import dgl 
import numpy as np
import dgl
import dgl.function as fn
from dgl import DGLGraph
import datetime
import math

# *******************************************
# Q(s,a) 
# *******************************************

def create_model(
	layer_sizes: Tuple,
	weight_init: str = "xavier_uniform",
	activation_func: str = "leaky_relu"
	):

	layers = []
	limit = len(layer_sizes)

	# add more activations
	if activation_func == "tanh":
		activation = nn.Tanh()
	elif activation_func == "relu":
		activation = nn.ReLU()
	elif activation_func == "leaky_relu":
		activation = nn.LeakyReLU()

	# add more weight init
	if weight_init == "xavier_uniform":
		weight_init = torch.nn.init.xavier_uniform_
	elif weight_init == "xavier_normal":
		weight_init = torch.nn.init.xavier_normal_

	for layer in range(limit - 1):
		act = activation if layer < limit - 2 else nn.Identity()
		layers += [nn.Linear(layer_sizes[layer], layer_sizes[layer + 1])]
		weight_init(layers[-1].weight)
		layers += [act]

	return nn.Sequential(*layers)




class ScalarDotProductCriticNetwork(nn.Module):
	def __init__(self, obs_act_input_dim, obs_act_output_dim, final_input_dim, final_output_dim, num_agents, num_actions, threshold=0.1):
		super(ScalarDotProductCriticNetwork, self).__init__()
		
		self.num_agents = num_agents
		self.num_actions = num_actions
		# self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		self.device = "cpu"

		self.key_layer = nn.Linear(obs_act_input_dim, obs_act_output_dim, bias=False)

		self.query_layer = nn.Linear(obs_act_input_dim, obs_act_output_dim, bias=False)

		self.attention_value_layer = nn.Linear(obs_act_input_dim, obs_act_output_dim, bias=False)

		# dimesion of key
		self.d_k_obs_act = obs_act_output_dim

		# NOISE
		self.noise_normal = torch.distributions.Normal(loc=torch.tensor([0.0]), scale=torch.tensor([1.0]))
		self.noise_uniform = torch.rand
		# ********************************************************************************************************

		# ********************************************************************************************************
		# FCN FINAL LAYER TO GET VALUES
		self.final_value_layer_1 = nn.Linear(final_input_dim, 64, bias=False)
		self.final_value_layer_2 = nn.Linear(64, final_output_dim, bias=False)
		# ********************************************************************************************************	

		self.place_policies = torch.zeros(self.num_agents,self.num_agents,obs_act_input_dim).to(self.device)
		self.place_actions = torch.ones(self.num_agents,self.num_agents,obs_act_input_dim).to(self.device)
		one_hots = torch.ones(obs_act_input_dim)
		zero_hots = torch.zeros(obs_act_input_dim)

		for j in range(self.num_agents):
			self.place_policies[j][j] = one_hots
			self.place_actions[j][j] = zero_hots

		self.threshold = threshold
		# ********************************************************************************************************* 

		self.reset_parameters()

	def reset_parameters(self):
		"""Reinitialize learnable parameters."""
		gain = nn.init.calculate_gain('leaky_relu')

		nn.init.xavier_uniform_(self.key_layer.weight)
		nn.init.xavier_uniform_(self.query_layer.weight)
		nn.init.xavier_uniform_(self.attention_value_layer.weight)


		nn.init.xavier_uniform_(self.final_value_layer_1.weight, gain=gain)
		nn.init.xavier_uniform_(self.final_value_layer_2.weight, gain=gain)



	def forward(self, states, policies, actions):

		# input to KEY, QUERY and ATTENTION VALUE NETWORK
		obs_actions = torch.cat([states,actions],dim=-1)
		# print("OBSERVATIONS ACTIONS")
		# print(obs_actions)

		# For calculating the right advantages
		obs_policy = torch.cat([states,policies], dim=-1)

		# KEYS
		key_obs_actions = self.key_layer(obs_actions)

		# QUERIES
		query_obs_actions = self.query_layer(obs_actions)

		# SCORE
		score_obs_actions = torch.mm(query_obs_actions,key_obs_actions.transpose(0,1)).transpose(0,1).reshape(-1,1)
		score_obs_actions = score_obs_actions.reshape(-1,self.num_agents,1)
		weight = F.softmax(score_obs_actions/math.sqrt(self.d_k_obs_act), dim=-2)
		ret_weight = weight
		
		obs_actions = obs_actions.repeat(self.num_agents,1).reshape(self.num_agents,self.num_agents,-1)
		obs_policy = obs_policy.repeat(self.num_agents,1).reshape(self.num_agents,self.num_agents,-1)
		obs_actions_policies = self.place_policies*obs_policy + self.place_actions*obs_actions

		attention_values = torch.tanh(self.attention_value_layer(obs_actions_policies))

		current_node_states = states.unsqueeze(-2).repeat(1,self.num_agents,1)

		attention_values = attention_values.repeat(self.num_agents,1,1).reshape(self.num_agents,self.num_agents,self.num_agents,-1)
		# SOFTMAX
		weight = weight.repeat(1,self.num_agents,1).reshape(self.num_agents,self.num_agents,self.num_agents,1)
		weighted_attention_values = attention_values*weight

		# SOFTMAX WITH NOISE
		# weight = weight.unsqueeze(-2).repeat(1,1,self.num_agents,1).unsqueeze(-1)
		# uniform_noise = (self.noise_uniform((attention_values.view(-1).size())).reshape(attention_values.size()) - 0.5) * 0.1 #SCALING NOISE AND MAKING IT ZERO CENTRIC
		# weighted_attention_values = attention_values*weight + uniform_noise

		# SOFTMAX WITH NORMALIZATION
		# scaling_weight = F.relu(weight - self.threshold)
		# scaling_weight = torch.div(scaling_weight,torch.sum(scaling_weight,dim =-1).unsqueeze(-1))
		# ret_weight = scaling_weight
		# scaling_weight = scaling_weight.unsqueeze(-2).repeat(1,1,self.num_agents,1).unsqueeze(-1)
		# weighted_attention_values = attention_values*scaling_weight

		node_features = torch.mean(weighted_attention_values, dim=-2)

		Value = F.leaky_relu(self.final_value_layer_1(node_features))
		Value = self.final_value_layer_2(Value)

		return Value, ret_weight


class PolicyNetwork(nn.Module):
	def __init__(
		self,
		policy_sizes
		):
		super(PolicyNetwork,self).__init__()

		self.policy = create_model(policy_sizes)

	def forward(self,states):
		return F.softmax(self.policy(states),-1)





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

	return states_critic, states_actor


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




def calculate_indiv_weights(weights, weight_dictionary, num_agents):

	for i in range(num_agents):
		agent_name = 'agent %d' % i
		for j in range(num_agents):
			agent_name_ = 'agent %d' % j
			weight_dictionary[agent_name][agent_name_] = weights[i][j].item()



def run(env, max_steps):  


	# cut the tail of softmax --> Used in softmax with normalization
	softmax_cut_threshold = 0.1

	device = "cpu"

	num_agents = env.n
	num_actions = env.action_space[0].n

	obs_input_dim = 2*3
	obs_act_input_dim = obs_input_dim + num_actions # (pose,vel,goal pose, paired agent goal pose) --> observations 
	obs_act_output_dim = 16
	final_input_dim = obs_act_output_dim
	final_output_dim = 1

	# SCALAR DOT PRODUCT
	critic_network = ScalarDotProductCriticNetwork(obs_act_input_dim, obs_act_output_dim, final_input_dim, final_output_dim, num_agents, num_actions, softmax_cut_threshold).to(device)

	policy_input_dim = 2*3 + (num_agents-1)*2*2 # pose,vel,goal --> itself; relative pose and vel --> other agents
	policy_output_dim = env.action_space[0].n
	policy_network_size = (policy_input_dim,512,256,policy_output_dim)
	policy_network = PolicyNetwork(policy_network_size).to(device)


	# Loading models
	model_path_value = "../../../models/Scalar_dot_product/collision_avoidance/4_Agents/SingleAttentionMechanism/with_prd_soft_adv/critic_networks/14-05-2021VN_ATN_FCN_lr0.01_PN_FCN_lr0.0002_GradNorm0.5_Entropy0.008_trace_decay0.98lambda_0.001topK_2select_above_threshold0.1softmax_cut_threshold0.1_epsiode29000.pt"
	model_path_policy = "../../../models/Scalar_dot_product/collision_avoidance/4_Agents/SingleAttentionMechanism/with_prd_soft_adv/actor_networks/14-05-2021_PN_FCN_lr0.0002VN_SAT_FCN_lr0.01_GradNorm0.5_Entropy0.008_trace_decay0.98lambda_0.001topK_2select_above_threshold0.1softmax_cut_threshold0.1_epsiode29000.pt"
	# For CPU
	critic_network.load_state_dict(torch.load(model_path_value,map_location=torch.device('cpu')))
	policy_network.load_state_dict(torch.load(model_path_policy,map_location=torch.device('cpu')))
	# For GPU
	# critic_network.load_state_dict(torch.load(model_path_value))
	# policy_network.load_state_dict(torch.load(model_path_policy))

	tensorboard_dir = '../../../runs/plot_weights_gif_collision_avoidance/Scalar_dot_product/collision_avoidance/4_Agents/SingleAttentionMechanism/without_prd/'
	writer = SummaryWriter(tensorboard_dir)


	gif_dir = '../../../gifs/plot_weights_gif_collision_avoidance/Scalar_dot_product/collision_avoidance/4_Agents/SingleAttentionMechanism/without_prd/'
	try: 
		os.makedirs(gif_dir, exist_ok = True) 
		print("Gif Directory created successfully") 
	except OSError as error: 
		print("Gif Directory can not be created")

	gif_path = gif_dir+'collision_avoidance_without_prd.gif'

	weight_dictionary = {}

	for i in range(num_agents):
		agent_name = 'agent %d' % i
		weight_dictionary[agent_name] = {}
		for j in range(num_agents):
			agent_name_ = 'agent %d' % j
			weight_dictionary[agent_name][agent_name_] = 0


	states = env.reset()

	images = []

	states_critic,states_actor = split_states(states, num_agents)

	for step in range(max_steps):

		# At each step, append an image to list
		images.append(np.squeeze(env.render(mode='rgb_array')))
		
		actions = []
		with torch.no_grad():
			probabilities = []
			for i in range(num_agents):
				dists = policy_network.forward(torch.FloatTensor(states_actor[i]).to(device))
				probs = Categorical(dists)
				action = probs.sample().cpu().detach().item()
				actions.append(action)
				probabilities.append(dists)

			one_hot_actions = np.zeros((num_agents,num_actions))
			for i,act in enumerate(actions):
				one_hot_actions[i][act] = 1

			V_values, weights = critic_network.forward(torch.FloatTensor(states_critic).to(device), torch.stack(probabilities).to(device), torch.FloatTensor(one_hot_actions).to(device))


		# Advance a step and render a new image
		next_states,rewards,dones,info = env.step(actions)
		next_states_critic,next_states_actor = split_states(next_states, num_agents)

		total_rewards = np.sum(rewards)

		print("*"*100)
		print("TIMESTEP: {} | REWARD: {} \n".format(step,np.round(total_rewards,decimals=4)))
		print("*"*100)


		states_critic,states_actor = next_states_critic,next_states_actor
		states = next_states

		writer.add_scalar('Reward Incurred/Reward',total_rewards,step)
		calculate_indiv_weights(weights, weight_dictionary,num_agents)
		for i in range(num_agents):
			agent = 'agent %d' % i
			writer.add_scalars('Weights/Average_Weights/'+agent,weight_dictionary[agent],step)
		writer.add_scalar('Reward Incurred/Reward',total_rewards,step)


		


	print("GENERATING GIF")
	make_gif(np.array(images),gif_path)


if __name__ == '__main__':
	env = make_env(scenario_name="collision_avoidance",benchmark=False)

	run(env,50)