import pygame
import sys
import random
import time
from a2c_model import *
from multiagent.environment import MultiAgentEnv
# from multiagent.scenarios.simple_spread import Scenario
import multiagent.scenarios as scenarios
import torch
from torch.distributions import Categorical
import numpy as np

num_actions = 5
num_agents = 8
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TITLE = "SOCIAL DILEMMA" # PAIRED AGENT, CROSSING GREEDY (v1), CROSSING FULLY COOPERATIVE (v2), CROSSING PARTIALLY COOPERATIVE (v3), SOCIAL DILEMMA
ENVIRONMENT_NAME = "color_social_dilemma" # paired_by_sharing_goals, crossing_fully_coop, crossing_greedy, crossing_partially_coop, color_social_dilemma
SCREEN_SIZE = WIDTH, HEIGHT = (800, 800)
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 50, 50)
GREEN = (50, 255, 50)
AGENT_RADIUS = 800/20
LANDMARK_RADIUS = 800/40


if ENVIRONMENT_NAME == "paired_by_sharing_goals":
	obs_dim = 2*4
	critic_network = GATCritic(obs_dim, 128, obs_dim+num_actions, 128, 128, 1, num_agents, num_actions).to(device)
elif ENVIRONMENT_NAME == "crossing_greedy":
	obs_dim = 2*3 + (num_agents-1)*2
	critic_network = GATCritic(obs_dim, 128, obs_dim+num_actions, 128, 128, 1, num_agents, num_actions).to(device)
elif ENVIRONMENT_NAME == "crossing_fully_coop":
	obs_dim = 2*3
	critic_network = DualGATCritic(obs_dim, 128, obs_dim+num_actions, 128, 128, 1, num_agents, num_actions).to(device)
elif ENVIRONMENT_NAME == "color_social_dilemma":
	obs_dim = 2*2 + 1 + 2*3
	critic_network = GATCritic(obs_dim, 128, obs_dim+num_actions, 128, 128, 1, num_agents, num_actions).to(device)
elif ENVIRONMENT_NAME == "crossing_partially_coop":
	obs_dim = 2*3 + 1
	critic_network = DualGATCritic(obs_dim, 128, obs_dim+num_actions, 128, 128, 1, num_agents, num_actions).to(device)


if ENVIRONMENT_NAME in ["paired_by_sharing_goals", "crossing_greedy", "crossing_fully_coop"]:
	obs_dim = 2*3
elif ENVIRONMENT_NAME in ["color_social_dilemma"]:
	obs_dim = 2*2 + 1 + 2*3
elif ENVIRONMENT_NAME in ["crossing_partially_coop"]:
	obs_dim = 2*3 + 1

# MLP POLICY
policy_network = MLPPolicyNetwork(obs_dim, num_agents, num_actions).to(device)


# Loading models

# CROSSING GREEDY WITH 8 AGENTS
if ENVIRONMENT_NAME == "crossing_greedy":
	model_path_value = "../../../tests/policy_eval/crossing_8_agents_pen_colliding_agents_policy_eval/models/crossing_prd_above_threshold_run2/critic_networks/17-08-2021VN_ATN_FCN_lr0.001_PN_ATN_FCN_lr0.0005_GradNorm0.5_Entropy0.008_trace_decay0.98topK_0select_above_threshold0.01l1_pen0.0critic_entropy_pen0.0_epsiode20000.pt"
	model_path_policy = "../../../tests/policy_eval/crossing_8_agents_pen_colliding_agents_policy_eval/models/crossing_prd_above_threshold_run2/actor_networks/17-08-2021_PN_ATN_FCN_lr0.0005VN_SAT_FCN_lr0.001_GradNorm0.5_Entropy0.008_trace_decay0.98topK_0select_above_threshold0.01l1_pen0.0critic_entropy_pen0.0_epsiode20000.pt"
	# # For CPU
	# # critic_network.load_state_dict(torch.load(model_path_value,map_location=torch.device('cpu')))
	# # policy_network.load_state_dict(torch.load(model_path_policy,map_location=torch.device('cpu')))
	# # # For GPU
	critic_network.load_state_dict(torch.load(model_path_value))
	policy_network.load_state_dict(torch.load(model_path_policy))

# PAIRED AGENT WITH 8 AGENTS
elif ENVIRONMENT_NAME == "paired_by_sharing_goals":
	model_path_value = "../../../tests/policy_eval/coma_to_prd/coma_v6/models/paired_by_sharing_goals_prd_above_threshold_ascend_run1/critic_networks/24-08-2021VN_ATN_FCN_lr0.001_PN_ATN_FCN_lr0.0005_GradNorm0.5_Entropy0.008_trace_decay0.98topK_0select_above_threshold0.0l1_pen0.0critic_entropy_pen0.0_epsiode200000.pt"
	model_path_policy = "../../../tests/policy_eval/coma_to_prd/coma_v6/models/paired_by_sharing_goals_prd_above_threshold_ascend_run1/actor_networks/24-08-2021_PN_ATN_FCN_lr0.0005VN_SAT_FCN_lr0.001_GradNorm0.5_Entropy0.008_trace_decay0.98topK_0select_above_threshold0.0l1_pen0.0critic_entropy_pen0.0_epsiode200000.pt"
	# For CPU
	# critic_network.load_state_dict(torch.load(model_path_value,map_location=torch.device('cpu')))
	# policy_network.load_state_dict(torch.load(model_path_policy,map_location=torch.device('cpu')))
	# # For GPU
	critic_network.load_state_dict(torch.load(model_path_value))
	policy_network.load_state_dict(torch.load(model_path_policy))

# CROSSING FULLY COOP WITH 8 AGENTS
elif ENVIRONMENT_NAME == "crossing_fully_coop":
	model_path_value = "../../../tests/policy_eval/crossing_8_agents_pen_non_colliding_team_members_DUALGAT_policy_eval/models/crossing_prd_above_threshold_ascend_run3/critic_networks/25-08-2021VN_ATN_FCN_lr0.001_PN_ATN_FCN_lr0.0005_GradNorm0.5_Entropy0.008_trace_decay0.98topK_0select_above_threshold0.0l1_pen0.0critic_entropy_pen0.0_epsiode200000.pt"
	model_path_policy = "../../../tests/policy_eval/crossing_8_agents_pen_non_colliding_team_members_DUALGAT_policy_eval/models/crossing_prd_above_threshold_ascend_run3/actor_networks/25-08-2021_PN_ATN_FCN_lr0.0005VN_SAT_FCN_lr0.001_GradNorm0.5_Entropy0.008_trace_decay0.98topK_0select_above_threshold0.0l1_pen0.0critic_entropy_pen0.0_epsiode200000.pt"
	# For CPU
	# critic_network.load_state_dict(torch.load(model_path_value,map_location=torch.device('cpu')))
	# policy_network.load_state_dict(torch.load(model_path_policy,map_location=torch.device('cpu')))
	# # For GPU
	critic_network.load_state_dict(torch.load(model_path_value))
	policy_network.load_state_dict(torch.load(model_path_policy))

# CROSSING PARTIAL COOP WITH 16 AGENTS
elif ENVIRONMENT_NAME == "crossing_partially_coop":
	model_path_value = "../../../tests/policy_eval/team_crossing_16_agents_pen_non_colliding_team_members_DUALGAT_policy_eval/models/team_crossing_prd_above_threshold_ascend_run1/critic_networks/24-08-2021VN_ATN_FCN_lr0.001_PN_ATN_FCN_lr0.0005_GradNorm0.5_Entropy0.008_trace_decay0.98topK_0select_above_threshold0.0l1_pen0.0critic_entropy_pen0.0_epsiode200000.pt"
	model_path_policy = "../../../tests/policy_eval/team_crossing_16_agents_pen_non_colliding_team_members_DUALGAT_policy_eval/models/team_crossing_prd_above_threshold_ascend_run1/actor_networks/24-08-2021_PN_ATN_FCN_lr0.0005VN_SAT_FCN_lr0.001_GradNorm0.5_Entropy0.008_trace_decay0.98topK_0select_above_threshold0.0l1_pen0.0critic_entropy_pen0.0_epsiode200000.pt"
	# For CPU
	# critic_network.load_state_dict(torch.load(model_path_value,map_location=torch.device('cpu')))
	# policy_network.load_state_dict(torch.load(model_path_policy,map_location=torch.device('cpu')))
	# # For GPU
	critic_network.load_state_dict(torch.load(model_path_value))
	policy_network.load_state_dict(torch.load(model_path_policy))

# SOCIAL DILEMMA WITH 8 AGENTS
elif ENVIRONMENT_NAME == "color_social_dilemma":
	model_path_value = "../../../tests/policy_eval/color_social_dilemma_8_Agents_50K_policy_eval/models/color_social_dilemma_pt2_prd_above_threshold_ascend_run4/critic_networks/19-08-2021VN_ATN_FCN_lr0.001_PN_ATN_FCN_lr0.0001_GradNorm0.5_Entropy0.008_trace_decay0.98topK_0select_above_threshold0.0l1_pen0.0critic_entropy_pen0.0_epsiode50000.pt"
	model_path_policy = "../../../tests/policy_eval/color_social_dilemma_8_Agents_50K_policy_eval/models/color_social_dilemma_pt2_prd_above_threshold_ascend_run4/actor_networks/19-08-2021_PN_ATN_FCN_lr0.0001VN_SAT_FCN_lr0.001_GradNorm0.5_Entropy0.008_trace_decay0.98topK_0select_above_threshold0.0l1_pen0.0critic_entropy_pen0.0_epsiode50000.pt"
	# For CPU
	# critic_network.load_state_dict(torch.load(model_path_value,map_location=torch.device('cpu')))
	# policy_network.load_state_dict(torch.load(model_path_policy,map_location=torch.device('cpu')))
	# # For GPU
	critic_network.load_state_dict(torch.load(model_path_value))
	policy_network.load_state_dict(torch.load(model_path_policy))

# Initialization
pygame.init()
screen = pygame.display.set_mode(SCREEN_SIZE)
pygame.display.set_caption(TITLE)
fps = pygame.time.Clock()
paused = False

agent_position = []
landmark_position = []
if ENVIRONMENT_NAME in ["crossing_partially_coop", "color_social_dilemma"]:
	colors = [np.array([255,0,0]), np.array([0,255,0]), np.array([0,0,255])]
	agent_color = [colors[0]]*(num_agents//2) + [colors[1]]*(num_agents//2)
else:
	agent_color = [random.sample(range(0, 255), 3) for i in range(num_agents//2)]
	agent_color = agent_color+agent_color[::-1]
no_act = 0
left_act = 1
right_act = 2
down_act = 3
up_act = 4

# (-1,1) --> (0,HEIGHT) (0,WIDTH)
def scale_pose_to_screen(x,y):
	posex = (x+1)*WIDTH/2
	posey = (y+1)*HEIGHT/2
	return posex, posey

# Ball setup
def init_pose(states):
	for i in range(num_agents):
		posex, posey = scale_pose_to_screen(states[i][0], states[i][1])
		agent_position.append([posex,posey])
		if ENVIRONMENT_NAME == "color_social_dilemma":
			if i>0:
				continue
			else:
				posex, posey = scale_pose_to_screen(states[i][5], states[i][6])
				landmark_position.append([posex,posey])
				posex, posey = scale_pose_to_screen(states[i][8], states[i][9])
				landmark_position.append([posex,posey])
		else:
			posex, posey = scale_pose_to_screen(states[i][4], states[i][5])
			landmark_position.append([posex,posey])


def map_weight_to_width(weight):
	# divide into 10 buckets, weight --> (0,1)
	# return round(weight*10)
	if weight>0.01:
		return 5
	else:
		return 0

def update(states):
	for i in range(num_agents):
		pose_x, pose_y = scale_pose_to_screen(states[i][0],states[i][1])
		agent_position[i][0], agent_position[i][1] = pose_x, pose_y


def render(states_critic, weights):
	screen.fill(BLACK)

	for i in range(num_agents):
		pygame.draw.circle(screen, agent_color[i], agent_position[i], AGENT_RADIUS, 0)
		if ENVIRONMENT_NAME == "color_social_dilemma":
			if i>1:
				continue
			elif i==0:
				pygame.draw.circle(screen, agent_color[0], landmark_position[i], LANDMARK_RADIUS*4, 0)
			elif i==1:
				pygame.draw.circle(screen, agent_color[-1], landmark_position[i], LANDMARK_RADIUS*4, 0)
		else:
			pygame.draw.circle(screen, agent_color[i], landmark_position[i], LANDMARK_RADIUS, 0)
		if i == 0:
			for j in range(num_agents):
				pygame.draw.line(screen, agent_color[i], agent_position[i], agent_position[j], map_weight_to_width(weights[i][j].item()))

	pygame.display.update()
	fps.tick(60)





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



def run(env, max_steps):

	paused = False
	states = env.reset()
	states_critic,states_actor = split_states(states, num_agents)

	init_pose(states_critic)

	for step in range(max_steps):
		
		actions = []
		with torch.no_grad():
			states_actor_ = torch.FloatTensor([states_actor]).to(device)
			dists, _ = policy_network.forward(states_actor_)
			# sample an action
			actions = [Categorical(dist).sample().cpu().detach().item() for dist in dists[0]]
			# choose best action only
			# actions = [torch.argmax(dist).cpu().detach().item() for dist in dists[0]]

			one_hot_actions = np.zeros((num_agents,num_actions))
			for i,act in enumerate(actions):
				one_hot_actions[i][act] = 1

			if ENVIRONMENT_NAME in ["crossing_partially_coop", "crossing_fully_coop"]:
				_, weights_preproc, weights_postproc = critic_network.forward(torch.FloatTensor([states_critic]).to(device), dists, torch.FloatTensor([one_hot_actions]).to(device))
				weights_prd = (weights_preproc+weights_postproc)/2.0
			else:
				_, weights = critic_network.forward(torch.FloatTensor([states_critic]).to(device), dists, torch.FloatTensor([one_hot_actions]).to(device))
				weights_prd = weights
		
		# Advance a step and render a new image
		next_states,rewards,dones,info = env.step(actions)
		next_states_critic,next_states_actor = split_states(next_states, num_agents)

		total_rewards = np.sum(rewards)

		print("*"*100)
		print("TIMESTEP: {} | REWARD: {} \n".format(step,np.round(total_rewards,decimals=4)))
		print("*"*100)


		for event in pygame.event.get():
			if event.type == pygame.QUIT:
				pygame.quit()
				sys.exit()
			if event.type == pygame.KEYUP:
				if event.key == pygame.K_SPACE:
					paused = not paused
		if not paused:
			update(next_states_critic)
			render(states_critic, weights_prd[0])


		states_critic,states_actor = next_states_critic,next_states_actor
		states = next_states
		time.sleep(0.5)


if __name__ == '__main__':
	env = make_env(scenario_name=ENVIRONMENT_NAME,benchmark=False)
	run(env,100)