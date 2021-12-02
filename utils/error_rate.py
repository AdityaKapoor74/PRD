from a2c_model import *
import torch
from torch.distributions import Categorical

from multiagent.environment import MultiAgentEnv
import multiagent.scenarios as scenarios
import numpy as np
import os

def make_env(scenario_name, benchmark=False):
	scenario = scenarios.load(scenario_name + ".py").Scenario()
	world = scenario.make_world()
	if benchmark:
		env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation, scenario.benchmark_data, scenario.isFinished)
	else:
		env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation, None, scenario.isFinished)
	return env


env_name = "crossing_partially_coop"
env = make_env(scenario_name=env_name,benchmark=False)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

num_agents = env.n
num_actions = env.action_space[0].n

if env_name == "paired_by_sharing_goals":
	obs_dim = 2*4
	critic_network = TransformerCritic(obs_dim, 128, obs_dim+num_actions, 128, 128, 1, num_agents, num_actions, device).to(device)
	epsiode_list = [i for i in range(1000,20001,1000)]
elif env_name == "crossing_greedy":
	obs_dim = 2*3
	critic_network = TransformerCritic(obs_dim, 128, obs_dim+num_actions, 128, 128, 1, num_agents, num_actions, device).to(device)
	epsiode_list = [i for i in range(1000,20001,1000)]
elif env_name == "crossing_fully_coop":
	obs_dim = 2*3
	critic_network = DualTransformerCritic(obs_dim, 128, obs_dim+num_actions, 128, 128, 1, num_agents, num_actions, device).to(device)
	epsiode_list = [i for i in range(1000,200001,1000)]
elif env_name == "color_social_dilemma":
	obs_dim = 2*2 + 1 + 2*3
	critic_network = TransformerCritic(obs_dim, 128, obs_dim+num_actions, 128, 128, 1, num_agents, num_actions, device).to(device)
	epsiode_list = [i for i in range(1000,50001,1000)]
elif env_name in ["crossing_partially_coop", "crossing_team_greedy"]:
	obs_dim = 2*3 + 1
	critic_network = DualTransformerCritic(obs_dim, 128, obs_dim+num_actions, 128, 128, 1, num_agents, num_actions, device).to(device)
	epsiode_list = [i for i in range(1000,200001,1000)]

if env_name in ["paired_by_sharing_goals", "crossing_greedy", "crossing_fully_coop"]:
	obs_dim = 2*3
elif env_name in ["color_social_dilemma"]:
	obs_dim = 2*2 + 1 + 2*3
elif env_name in ["crossing_partially_coop", "crossing_team_greedy"]:
	obs_dim = 2*3 + 1

# MLP POLICY
policy_network = MLPPolicy(obs_dim, num_agents, num_actions, device).to(device)

model_value = "../../../tests/policy_eval/crossing_partially_coop_24_agents_3_teams/models/crossing_partially_coop_shared_run1/critic_networks/02-09-2021VN_ATN_FCN_lr0.001_PN_ATN_FCN_lr0.0005_GradNorm0.5_Entropy0.008_trace_decay0.98topK_0select_above_threshold0.0l1_pen0.0critic_entropy_pen0.0_epsiode"
model_policy = "../../../tests/policy_eval/crossing_partially_coop_24_agents_3_teams/models/crossing_partially_coop_shared_run1/actor_networks/02-09-2021_PN_ATN_FCN_lr0.0005VN_SAT_FCN_lr0.001_GradNorm0.5_Entropy0.008_trace_decay0.98topK_0select_above_threshold0.0l1_pen0.0critic_entropy_pen0.0_epsiode"


relevant_set = None
if env_name == "paired_by_sharing_goals":
	relevant_set = torch.ones(num_agents,num_agents).to(device)
	for i in range(num_agents):
		relevant_set[i][num_agents-i-1] = 0

	# here the relevant set is given value=0
	relevant_set = torch.transpose(relevant_set,0,1)
elif env_name == "crossing_partially_coop":
	team_size = 8
	relevant_set = torch.ones(num_agents,num_agents).to(device)
	for i in range(num_agents):
		for j in range(num_agents):
			if i<team_size and j<team_size:
				relevant_set[i][j] = 0
			elif i>=team_size and i<2*team_size and j>=team_size and j<2*team_size:
				relevant_set[i][j] = 0
			elif i>=2*team_size and i<3*team_size and j>=2*team_size and j<3*team_size:
				relevant_set[i][j] = 0
			else:
				break

	# here the relevant set is given value=0
	relevant_set = torch.transpose(relevant_set,0,1)


def split_states(states):
	states_critic = []
	states_actor = []
	for i in range(num_agents):
		states_critic.append(states[i][0])
		states_actor.append(states[i][1])

	states_critic = np.asarray(states_critic)
	states_actor = np.asarray(states_actor)

	return states_critic,states_actor


def calculate_prd_weights(weights, critic_name):
	# print("weights", weights[0].shape)
	weights_prd = None
	if "MultiHeadDual" in critic_name:
		weights_ = torch.stack([weight for weight in weights[1]])
		weights_prd = torch.mean(weights_, dim=0)
	elif "MultiHead" in critic_name:
		weights_ = torch.stack([weight for weight in weights[0]])
		weights_prd = torch.mean(weights_, dim=0)
	elif "Dual" in critic_name:
		weights_prd = weights[1]
	else:
		weights_prd = weights[0]

	return weights_prd


def get_action(state):
	state = torch.FloatTensor([state]).to(device)
	dists, _ = policy_network.forward(state)
	index = [Categorical(dist).sample().cpu().detach().item() for dist in dists[0]]
	return index, dists



error_rate = []
relevant_set_size = []
select_above_threshold = 0.04
max_time_steps = 100

for eps in epsiode_list:

	if torch.cuda.is_available() is False:
		# For CPU
		critic_network.load_state_dict(torch.load(model_value+str(eps)+".pt",map_location=torch.device('cpu')))
		policy_network.load_state_dict(torch.load(model_policy+str(eps)+".pt",map_location=torch.device('cpu')))
	else:
		# For GPU
		critic_network.load_state_dict(torch.load(model_value+str(eps)+".pt"))
		policy_network.load_state_dict(torch.load(model_policy+str(eps)+".pt"))

	states = env.reset()

	states_critic,states_actor = split_states(states)

	episode_reward = 0
	relevant_set_error_rate = 0
	avg_agent_group_over_episode = 0

	for step in range(1, max_time_steps+1):

		actions, probs = get_action(states_actor)
		one_hot_actions = np.zeros((num_agents,num_actions))
		for i,act in enumerate(actions):
			one_hot_actions[i][act] = 1

		one_hot_actions = torch.Tensor([one_hot_actions]).to(device)
		states_critic_ = torch.FloatTensor([states_critic]).to(device)
		Value_return = critic_network.forward(states_critic_, probs.detach(), one_hot_actions)
		V_values = Value_return[0]
		weights_value = Value_return[1:]

		weights_prd = calculate_prd_weights(weights_value, critic_network.name)

		masking_weights = (weights_prd>select_above_threshold).int()

		relevant_set_error_rate += torch.mean(masking_weights*relevant_set)


		avg_agent_group_over_episode = torch.sum(masking_weights.float(), dim=-2)
		avg_agent_group_over_episode = torch.mean(avg_agent_group_over_episode)

		next_states,rewards,dones,info = env.step(actions)
		next_states_critic,next_states_actor = split_states(next_states)


		if env_name in ["crossing_greedy", "crossing_fully_coop", "crossing_partially_coop", "crossing_team_greedy"]:
			collision_rate = [value[1] for value in rewards]
			rewards = [value[0] for value in rewards]

		episode_reward += np.sum(rewards)

		if all(dones) or step == max_time_steps:

			print("*"*100)
			print("EPISODE: {} | REWARD: {} | TIME TAKEN: {} / {} \n".format(eps,np.round(episode_reward,decimals=4),step,max_time_steps))
			print("*"*100)

			final_timestep = step

			break
		else:
			states_critic,states_actor = next_states_critic,next_states_actor
			states = next_states

	error_rate.append(relevant_set_error_rate.item())
	relevant_set_size.append(avg_agent_group_over_episode.item())

np.save(os.path.join(env_name+"_mean_error_rate"), np.array(error_rate), allow_pickle=True, fix_imports=True)
np.save(os.path.join(env_name+"_average_relevant_set"), np.array(relevant_set_size), allow_pickle=True, fix_imports=True)