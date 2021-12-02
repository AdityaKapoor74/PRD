from comet_ml import Experiment
# import pygame
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
num_agents = 12
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TITLE = "CROSSING TEAM GREEDY" # PAIRED AGENT, CROSSING GREEDY (v1), CROSSING FULLY COOPERATIVE (v2), CROSSING PARTIALLY COOPERATIVE (v3), SOCIAL DILEMMA
ENVIRONMENT_NAME = "crossing_team_greedy" # paired_by_sharing_goals, crossing_fully_coop, crossing_greedy, crossing_partially_coop, color_social_dilemma
WIDTH, HEIGHT = (1000, 1000)
SCREEN_SIZE = (WIDTH, HEIGHT)
START, END = (100,900)
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 50, 50)
GREEN = (50, 255, 50)
AGENT_RADIUS = 800/20
LANDMARK_RADIUS = 800/40

obs_dim = 2*3+1

critic_name = "NormalizedAttentionTransformerCritic"

comet_ml = Experiment("im5zK8gFkz6j07uflhc3hXk8I",project_name="Ablation Study")
dictionary = {
	"critic_name": critic_name,
	"env_name": ENVIRONMENT_NAME
}
comet_ml.log_parameters(dictionary)

critics = {
"TransformerCritic":TransformerCritic(obs_dim, 128, obs_dim+num_actions, 128, 128, 1, num_agents, num_actions).to(device),
"MultiHeadTransformerCritic2":MultiHeadTransformerCritic(obs_dim, 128, obs_dim+num_actions, 128, 128, 1, num_agents, num_actions, num_heads=2).to(device),
"MultiHeadTransformerCritic4":MultiHeadTransformerCritic(obs_dim, 128, obs_dim+num_actions, 128, 128, 1, num_agents, num_actions, num_heads=4).to(device),
"MultiHeadTransformerCritic8":MultiHeadTransformerCritic(obs_dim, 128, obs_dim+num_actions, 128, 128, 1, num_agents, num_actions, num_heads=8).to(device),
"DualTransformerCritic":DualTransformerCritic(obs_dim, 128, obs_dim+num_actions, 128, 128, 1, num_agents, num_actions).to(device),
"MultiHeadDualTransformerCritic2":MultiHeadDualTransformerCritic(obs_dim, 128, obs_dim+num_actions, 128, 128, 1, num_agents, num_actions, num_heads_preproc=2, num_heads_postproc=2).to(device),
"MultiHeadDualTransformerCritic4":MultiHeadDualTransformerCritic(obs_dim, 128, obs_dim+num_actions, 128, 128, 1, num_agents, num_actions, num_heads_preproc=4, num_heads_postproc=4).to(device),
"MultiHeadDualTransformerCritic8":MultiHeadDualTransformerCritic(obs_dim, 128, obs_dim+num_actions, 128, 128, 1, num_agents, num_actions, num_heads_preproc=8, num_heads_postproc=8).to(device),
"SemiHardAttnTransformerCritic8thweight":SemiHardAttnTransformerCritic(obs_dim, 128, obs_dim+num_actions, 128, 128, 1, num_agents, num_actions, kth_weight=8).to(device),
"SemiHardAttnTransformerCriticweightthreshold0.03":SemiHardAttnTransformerCritic(obs_dim, 128, obs_dim+num_actions, 128, 128, 1, num_agents, num_actions, weight_threshold=0.03).to(device),
# "MultiHeadSemiHardAttnTransformerCritic8thweight2":MultiHeadSemiHardAttnTransformerCritic(obs_dim, 128, obs_dim+num_actions, 128, 128, 1, num_agents, num_actions, kth_weight=8, num_heads=2).to(device),
# "MultiHeadSemiHardAttnTransformerCritic8thweight4":MultiHeadSemiHardAttnTransformerCritic(obs_dim, 128, obs_dim+num_actions, 128, 128, 1, num_agents, num_actions, kth_weight=8, num_heads=4).to(device),
"MultiHeadSemiHardAttnTransformerCritic8thweight8":MultiHeadSemiHardAttnTransformerCritic(obs_dim, 128, obs_dim+num_actions, 128, 128, 1, num_agents, num_actions, kth_weight=8, num_heads=8).to(device),
"MultiHeadSemiHardAttnTransformerCriticweightthreshold0.03_2":MultiHeadSemiHardAttnTransformerCritic(obs_dim, 128, obs_dim+num_actions, 128, 128, 1, num_agents, num_actions, weight_threshold=0.03, num_heads=2).to(device),
"MultiHeadSemiHardAttnTransformerCriticweightthreshold0.03_4":MultiHeadSemiHardAttnTransformerCritic(obs_dim, 128, obs_dim+num_actions, 128, 128, 1, num_agents, num_actions, weight_threshold=0.03, num_heads=4).to(device),
"MultiHeadSemiHardAttnTransformerCriticweightthreshold0.03_8":MultiHeadSemiHardAttnTransformerCritic(obs_dim, 128, obs_dim+num_actions, 128, 128, 1, num_agents, num_actions, weight_threshold=0.03, num_heads=8).to(device),
"GATCritic":GATCritic(obs_dim, 128, obs_dim+num_actions, 128, 128, 1, num_agents, num_actions).to(device),
# "MultiHeadGATCritic2":MultiHeadGATCritic(obs_dim, 128, obs_dim+num_actions, 128, 128, 1, num_agents, num_actions, num_heads=2).to(device),
"MultiHeadGATCritic4":MultiHeadGATCritic(obs_dim, 128, obs_dim+num_actions, 128, 128, 1, num_agents, num_actions, num_heads=4).to(device),
"MultiHeadGATCritic8":MultiHeadGATCritic(obs_dim, 128, obs_dim+num_actions, 128, 128, 1, num_agents, num_actions, num_heads=8).to(device),
"SemiHardGATCritic8thweight":SemiHardGATCritic(obs_dim, 128, obs_dim+num_actions, 128, 128, 1, num_agents, num_actions, kth_weight=8).to(device),
"SemiHardGATCriticweightthreshold0.03":SemiHardGATCritic(obs_dim, 128, obs_dim+num_actions, 128, 128, 1, num_agents, num_actions, weight_threshold=0.03).to(device),
"MultiHeadSemiHardAttnGATCritic8thweight2":SemiHardMultiHeadGATCritic(obs_dim, 128, obs_dim+num_actions, 128, 128, 1, num_agents, num_actions, kth_weight=8, num_heads=2).to(device),
"MultiHeadSemiHardAttnGATCritic8thweight4":SemiHardMultiHeadGATCritic(obs_dim, 128, obs_dim+num_actions, 128, 128, 1, num_agents, num_actions, kth_weight=8, num_heads=2).to(device),
"MultiHeadSemiHardAttnGATCritic8thweight8":SemiHardMultiHeadGATCritic(obs_dim, 128, obs_dim+num_actions, 128, 128, 1, num_agents, num_actions, kth_weight=8, num_heads=2).to(device),
"MultiHeadSemiHardAttnGATCriticweightthreshold0.03_2":SemiHardMultiHeadGATCritic(obs_dim, 128, obs_dim+num_actions, 128, 128, 1, num_agents, num_actions, weight_threshold=0.03, num_heads=2).to(device),
"MultiHeadSemiHardAttnGATCriticweightthreshold0.03_4":SemiHardMultiHeadGATCritic(obs_dim, 128, obs_dim+num_actions, 128, 128, 1, num_agents, num_actions, weight_threshold=0.03, num_heads=2).to(device),
"MultiHeadSemiHardAttnGATCriticweightthreshold0.03_8":SemiHardMultiHeadGATCritic(obs_dim, 128, obs_dim+num_actions, 128, 128, 1, num_agents, num_actions, weight_threshold=0.03, num_heads=2).to(device),
"GATV2Critic":GATV2Critic(obs_dim, 128, obs_dim+num_actions, 128, 128, 1, num_agents, num_actions).to(device),
"MultiHeadGATV2Critic2":MultiHeadGATV2Critic(obs_dim, 128, obs_dim+num_actions, 128, 128, 1, num_agents, num_actions, num_heads=2).to(device),
"MultiHeadGATV2Critic4":MultiHeadGATV2Critic(obs_dim, 128, obs_dim+num_actions, 128, 128, 1, num_agents, num_actions, num_heads=4).to(device),
"MultiHeadGATV2Critic8":MultiHeadGATV2Critic(obs_dim, 128, obs_dim+num_actions, 128, 128, 1, num_agents, num_actions, num_heads=8).to(device),
"SemiHardGATV2Critic8thweight":SemiHardGATV2Critic(obs_dim, 128, obs_dim+num_actions, 128, 128, 1, num_agents, num_actions, kth_weight=8).to(device),
"SemiHardGATV2Criticweightthreshold0.03":SemiHardGATV2Critic(obs_dim, 128, obs_dim+num_actions, 128, 128, 1, num_agents, num_actions, weight_threshold=0.03).to(device),
"MultiHeadSemiHardAttnGATV2Critic8thweight2":SemiHardMultiHeadGATV2Critic(obs_dim, 128, obs_dim+num_actions, 128, 128, 1, num_agents, num_actions, kth_weight=8, num_heads=2).to(device),
"MultiHeadSemiHardAttnGATV2Critic8thweight4":SemiHardMultiHeadGATV2Critic(obs_dim, 128, obs_dim+num_actions, 128, 128, 1, num_agents, num_actions, kth_weight=8, num_heads=4).to(device),
"MultiHeadSemiHardAttnGATV2Critic8thweight8":SemiHardMultiHeadGATV2Critic(obs_dim, 128, obs_dim+num_actions, 128, 128, 1, num_agents, num_actions, kth_weight=8, num_heads=8).to(device),
"MultiHeadSemiHardAttnGATV2Criticweightthreshold0.03_2":SemiHardMultiHeadGATV2Critic(obs_dim, 128, obs_dim+num_actions, 128, 128, 1, num_agents, num_actions, weight_threshold=0.03, num_heads=2).to(device),
"MultiHeadSemiHardAttnGATV2Criticweightthreshold0.03_4":SemiHardMultiHeadGATV2Critic(obs_dim, 128, obs_dim+num_actions, 128, 128, 1, num_agents, num_actions, weight_threshold=0.03, num_heads=4).to(device),
"MultiHeadSemiHardAttnGATV2Criticweightthreshold0.03_8":SemiHardMultiHeadGATV2Critic(obs_dim, 128, obs_dim+num_actions, 128, 128, 1, num_agents, num_actions, weight_threshold=0.03, num_heads=8).to(device),
"NormalizedAttentionTransformerCritic":NormalizedAttentionTransformerCritic(obs_dim, 128, obs_dim+num_actions, 128, 128, 1, num_agents, num_actions).to(device),
"MultiHeadNormalizedAttentionTransformerCritic2":MultiHeadNormalizedAttentionTransformerCritic(obs_dim, 128, obs_dim+num_actions, 128, 128, 1, num_agents, num_actions, num_heads=2).to(device),
"MultiHeadNormalizedAttentionTransformerCritic4":MultiHeadNormalizedAttentionTransformerCritic(obs_dim, 128, obs_dim+num_actions, 128, 128, 1, num_agents, num_actions, num_heads=4).to(device),
"MultiHeadNormalizedAttentionTransformerCritic8":MultiHeadNormalizedAttentionTransformerCritic(obs_dim, 128, obs_dim+num_actions, 128, 128, 1, num_agents, num_actions, num_heads=8).to(device),
"SemiHardNormalizedAttentionTransformerCritic":SemiHardNormalizedAttentionTransformerCritic(obs_dim, 128, obs_dim+num_actions, 128, 128, 1, num_agents, num_actions).to(device),
"SemiHardMultiHeadNormalizedAttentionTransformerCritic2":SemiHardMultiHeadNormalizedAttentionTransformerCritic(obs_dim, 128, obs_dim+num_actions, 128, 128, 1, num_agents, num_actions, num_heads=2).to(device),
"SemiHardMultiHeadNormalizedAttentionTransformerCritic4":SemiHardMultiHeadNormalizedAttentionTransformerCritic(obs_dim, 128, obs_dim+num_actions, 128, 128, 1, num_agents, num_actions, num_heads=4).to(device),
"SemiHardMultiHeadNormalizedAttentionTransformerCritic8":SemiHardMultiHeadNormalizedAttentionTransformerCritic(obs_dim, 128, obs_dim+num_actions, 128, 128, 1, num_agents, num_actions, num_heads=8).to(device),
}


obs_dim = 2*3 + 1
critic_network = critics[critic_name]

# # MLP POLICY
policy_network = MLPPolicy(obs_dim, num_agents, num_actions).to(device)

common = "../../../tests/ablation_study/"
path = "NormalizedAT/tests/"+critic_name+"/"
model_path_value = common+path+"models/crossing_team_greedy_shared_run1/critic_networks/"
model_path_policy = common+path+"models/crossing_team_greedy_shared_run1/actor_networks/"
rem_value_path = "-09-2021VN_ATN_FCN_lr0.001_PN_ATN_FCN_lr0.0001_GradNorm0.5_Entropy0.008_trace_decay0.98topK_0select_above_threshold0.0l1_pen0.0critic_entropy_pen0.0_epsiode90000.pt"
rem_policy_path = "-09-2021_PN_ATN_FCN_lr0.0001VN_SAT_FCN_lr0.001_GradNorm0.5_Entropy0.008_trace_decay0.98topK_0select_above_threshold0.0l1_pen0.0critic_entropy_pen0.0_epsiode90000.pt"


critic_paths = {
"TransformerCritic":"24",
"MultiHeadTransformerCritic2":"24",
"MultiHeadTransformerCritic4":"24",
"MultiHeadTransformerCritic8":"24",
"DualTransformerCritic":"24",
"MultiHeadDualTransformerCritic2":"24",
"MultiHeadDualTransformerCritic4":"24",
"MultiHeadDualTransformerCritic8":"24",
"SemiHardAttnTransformerCritic8thweight":"24",
"SemiHardAttnTransformerCriticweightthreshold0.03":"24",
# "MultiHeadSemiHardAttnTransformerCritic8thweight2":"-",
# "MultiHeadSemiHardAttnTransformerCritic8thweight4":"-",
"MultiHeadSemiHardAttnTransformerCritic8thweight8":"27",
"MultiHeadSemiHardAttnTransformerCriticweightthreshold0.03_2":"27",
"MultiHeadSemiHardAttnTransformerCriticweightthreshold0.03_4":"27",
"MultiHeadSemiHardAttnTransformerCriticweightthreshold0.03_8":"27",
"GATCritic":"28",
"MultiHeadGATCritic2":"28",
"MultiHeadGATCritic4":"28",
"MultiHeadGATCritic8":"28",
"SemiHardGATCritic8thweight":"28",
"SemiHardGATCriticweightthreshold0.03":"28",
"MultiHeadSemiHardAttnGATCritic8thweight2":"28",
"MultiHeadSemiHardAttnGATCritic8thweight4":"28",
"MultiHeadSemiHardAttnGATCritic8thweight8":"28",
"MultiHeadSemiHardAttnGATCriticweightthreshold0.03_2":"28",
"MultiHeadSemiHardAttnGATCriticweightthreshold0.03_4":"28",
"MultiHeadSemiHardAttnGATCriticweightthreshold0.03_8":"28",
"GATV2Critic":"29",
"MultiHeadGATV2Critic2":"29",
"MultiHeadGATV2Critic4":"29",
"MultiHeadGATV2Critic8":"29",
"SemiHardGATV2Critic8thweight":"29",
"SemiHardGATV2Criticweightthreshold0.03":"29",
"MultiHeadSemiHardAttnGATV2Critic8thweight2":"30",
"MultiHeadSemiHardAttnGATV2Critic8thweight4":"30",
"MultiHeadSemiHardAttnGATV2Critic8thweight8":"30",
"MultiHeadSemiHardAttnGATV2Criticweightthreshold0.03_2":"30",
"MultiHeadSemiHardAttnGATV2Criticweightthreshold0.03_4":"30",
"MultiHeadSemiHardAttnGATV2Criticweightthreshold0.03_8":"30",
"NormalizedAttentionTransformerCritic":"30",
"MultiHeadNormalizedAttentionTransformerCritic2":"",
"MultiHeadNormalizedAttentionTransformerCritic4":"",
"MultiHeadNormalizedAttentionTransformerCritic8":"",
"SemiHardNormalizedAttentionTransformerCritic":"",
"SemiHardMultiHeadNormalizedAttentionTransformerCritic2":"",
"SemiHardMultiHeadNormalizedAttentionTransformerCritic4":"",
"SemiHardMultiHeadNormalizedAttentionTransformerCritic8":"",
}


# Loading models
model_path_value = model_path_value + critic_paths[critic_name] + rem_value_path 
model_path_policy =  model_path_policy + critic_paths[critic_name] + rem_policy_path 
# For CPU
# critic_network.load_state_dict(torch.load(model_path_value,map_location=torch.device('cpu')))
# policy_network.load_state_dict(torch.load(model_path_policy,map_location=torch.device('cpu')))
# # For GPU
critic_network.load_state_dict(torch.load(model_path_value))
policy_network.load_state_dict(torch.load(model_path_policy))



# Initialization
# pygame.init()
# screen = pygame.display.set_mode(SCREEN_SIZE)
# pygame.display.set_caption(TITLE)
# fps = pygame.time.Clock()
# surface = pygame.Surface(SCREEN_SIZE, pygame.SRCALPHA)
paused = False

agent_position = []
agent_circle = []
landmark_position = []
'''
agent in question: dark color
team1: blue
team2: green
team3: red
relevant set: opaque
not in relevant set: transparent
'''
agent_color = [np.array([0,0,255,50])]*(num_agents//3) + [np.array([0,255,0,50])]*(num_agents//3) + [np.array([255,0,0,50])]*(num_agents//3)
landmark_color = [np.array([0,0,0,200])]*(num_agents)

no_act = 0
left_act = 1
right_act = 2
down_act = 3
up_act = 4

# (-1,1) --> (0,HEIGHT) (0,WIDTH)
def scale_pose_to_screen(x,y):
	# posex = (x+1)*WIDTH/2
	# posey = (y+1)*HEIGHT/2
	# print("x",x,"y",y)
	posex = START+(x+1)*(END-START)/2
	posey = START+(y+1)*(END-START)/2
	return posex, posey

# Ball setup
def init_pose(states):
	for i in range(num_agents):
		posex, posey = scale_pose_to_screen(states[i][0], states[i][1])
		agent_position.append([posex,posey])
		
		posex, posey = scale_pose_to_screen(states[i][5], states[i][6])
		landmark_position.append([posex,posey])


def map_weight_to_width(weight, agent_index):
	if weight>0.1:
		if agent_index != 0:
			agent_color[agent_index][3] = 150
			pygame.draw.circle(surface, agent_color[agent_index], agent_position[agent_index], AGENT_RADIUS, 0)
		else:
			pygame.draw.circle(surface, agent_color[agent_index], agent_position[agent_index], AGENT_RADIUS, 0)
			pygame.draw.circle(surface, np.array([0,0,0,200]), agent_position[agent_index], AGENT_RADIUS, 5)
		return 5
	else:
		if agent_index != 0:
			agent_color[agent_index][3] = 50
			pygame.draw.circle(surface, agent_color[agent_index], agent_position[agent_index], AGENT_RADIUS, 0)
		else:
			pygame.draw.circle(surface, agent_color[agent_index], agent_position[agent_index], AGENT_RADIUS, 0)
			pygame.draw.circle(surface, np.array([0,0,0,200]), agent_position[agent_index], AGENT_RADIUS, 5)
		return 0

def update(states):
	for i in range(num_agents):
		pose_x, pose_y = scale_pose_to_screen(states[i][0],states[i][1])
		agent_position[i][0], agent_position[i][1] = pose_x, pose_y


def render(states_critic, weights):
	screen.fill(WHITE)
	surface.fill((255,255,255,0))
	for i in range(num_agents):

		if i == 0:
			for j in range(num_agents):
				pygame.draw.line(surface, BLACK, agent_position[i], agent_position[j], map_weight_to_width(weights[j].item(), j))

		
		pygame.draw.circle(surface, landmark_color[i], landmark_position[i], LANDMARK_RADIUS, 0)

		

		
	screen.blit(surface, (0,0))
	pygame.display.update()
	fps.tick(10000)



def plot(weights, step):

	if "MultiHeadDual" in critic_name:
		weights_ = torch.stack([weight[0][0] for weight in weights[1]])
		weights_prd = torch.mean(weights_, dim=0)
	elif "MultiHead" in critic_name:
		weights_ = torch.stack([weight[0][0] for weight in weights[0]])
		weights_prd = torch.mean(weights_, dim=0)
	elif "Dual" in critic_name:
		weights_prd = weights[1][0][0]
	else:
		weights_prd = weights[0][0][0]


	# top4
	values, indices = torch.topk(weights_prd,k=4,dim=-1)
	right = 0
	wrong = 0
	for index in indices:
		if index>=0 and index<=3:
			right+=1
		else:
			wrong+=1
	error_rate = (abs(4 - right)/4) * 100
	comet_ml.log_metric('Error rate (top4)', error_rate, step)
	comet_ml.log_metric('Right (top4)', right, step)
	comet_ml.log_metric('Wrong (top4)', wrong, step)
	# threshold 0.3
	weights_prd_ = weights_prd - 0.3
	masks = weights_prd_>0
	right = 0
	wrong = 0
	for index, value in enumerate(masks):
		if value.item() is True:
			if index>=0 and index<=3:
				right+=1
			else:
				wrong+=1
	error_rate = (abs(4 - right)/4) * 100
	comet_ml.log_metric('Error rate (threshold0.3)', error_rate, step)
	comet_ml.log_metric('Right (threshold0.3)', right, step)
	comet_ml.log_metric('Wrong (threshold0.3)', wrong, step)
	# threshold 0.25
	weights_prd_ = weights_prd - 0.25
	masks = weights_prd_>0
	right = 0
	wrong = 0
	for index, value in enumerate(masks):
		if value.item() is True:
			if index>=0 and index<=3:
				right+=1
			else:
				wrong+=1
	error_rate = (abs(4 - right)/4) * 100
	comet_ml.log_metric('Error rate (threshold0.25)', error_rate, step)
	comet_ml.log_metric('Right (threshold0.25)', right, step)
	comet_ml.log_metric('Wrong (threshold0.25)', wrong, step)
	# threshold 0.2
	weights_prd_ = weights_prd - 0.2
	masks = weights_prd_>0
	right = 0
	wrong = 0
	for index, value in enumerate(masks):
		if value.item() is True:
			if index>=0 and index<=3:
				right+=1
			else:
				wrong+=1
	error_rate = (abs(4 - right)/4) * 100
	comet_ml.log_metric('Error rate (threshold0.2)', error_rate, step)
	comet_ml.log_metric('Right (threshold0.2)', right, step)
	comet_ml.log_metric('Wrong (threshold0.2)', wrong, step)
	# threshold 0.1
	weights_prd_ = weights_prd - 0.1
	masks = weights_prd_>0
	right = 0
	wrong = 0
	for index, value in enumerate(masks):
		if value.item() is True:
			if index>=0 and index<=3:
				right+=1
			else:
				wrong+=1
	error_rate = (abs(4 - right)/4) * 100
	comet_ml.log_metric('Error rate  (threshold0.1)', error_rate, step)
	comet_ml.log_metric('Right (threshold0.1)', right, step)
	comet_ml.log_metric('Wrong (threshold0.1)', wrong, step)
	# threshold 0.05
	weights_prd_ = weights_prd - 0.05
	masks = weights_prd_>0
	right = 0
	wrong = 0
	for index, value in enumerate(masks):
		if value.item() is True:
			if index>=0 and index<=3:
				right+=1
			else:
				wrong+=1
	error_rate = (abs(4 - right)/4) * 100
	comet_ml.log_metric('Error rate (threshold0.05)', error_rate, step)
	comet_ml.log_metric('Right (threshold0.05)', right, step)
	comet_ml.log_metric('Wrong (threshold0.05)', wrong, step)
	# threshold 0.03
	weights_prd_ = weights_prd - 0.03
	masks = weights_prd_>0
	right = 0
	wrong = 0
	for index, value in enumerate(masks):
		if value.item() is True:
			if index>=0 and index<=3:
				right+=1
			else:
				wrong+=1
	error_rate = (abs(4 - right)/4) * 100
	comet_ml.log_metric('Error rate (threshold0.03)', error_rate, step)
	comet_ml.log_metric('Right (threshold0.03)', right, step)
	comet_ml.log_metric('Wrong (threshold0.03)', wrong, step)
	# threshold 0.02
	weights_prd_ = weights_prd - 0.02
	masks = weights_prd_>0
	right = 0
	wrong = 0
	for index, value in enumerate(masks):
		if value.item() is True:
			if index>=0 and index<=3:
				right+=1
			else:
				wrong+=1
	error_rate = (abs(4 - right)/4) * 100
	comet_ml.log_metric('Error rate (threshold0.02)', error_rate, step)
	comet_ml.log_metric('Right (threshold0.02)', right, step)
	comet_ml.log_metric('Wrong (threshold0.02)', wrong, step)
	# threshold 0.01
	weights_prd_ = weights_prd - 0.01
	masks = weights_prd_>0
	right = 0
	wrong = 0
	for index, value in enumerate(masks):
		if value.item() is True:
			if index>=0 and index<=3:
				right+=1
			else:
				wrong+=1
	error_rate = (abs(4 - right)/4) * 100
	comet_ml.log_metric('Error rate (threshold0.01)', error_rate, step)
	comet_ml.log_metric('Right (threshold0.01)', right, step)
	comet_ml.log_metric('Wrong (threshold0.01)', wrong, step)

	return weights_prd

	





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
			# actions = [Categorical(dist).sample().cpu().detach().item() for dist in dists[0]]
			# choose best action only
			actions = [torch.argmax(dist).cpu().detach().item() for dist in dists[0]]

			one_hot_actions = np.zeros((num_agents,num_actions))
			for i,act in enumerate(actions):
				one_hot_actions[i][act] = 1

			V_ret = critic_network.forward(torch.FloatTensor([states_critic]).to(device), dists, torch.FloatTensor([one_hot_actions]).to(device))
			V_values = V_ret[0]
			weights = V_ret[1:]
			weights_prd = plot(weights, step)
		
		# Advance a step and render a new image
		next_states,rewards,dones,info = env.step(actions)
		next_states_critic,next_states_actor = split_states(next_states, num_agents)

		total_rewards = np.sum(rewards)

		print("*"*100)
		print("TIMESTEP: {} | REWARD: {} \n".format(step,np.round(total_rewards,decimals=4)))
		print("*"*100)


		# for event in pygame.event.get():
		# 	if event.type == pygame.QUIT:
		# 		pygame.quit()
		# 		sys.exit()
		# 	if event.type == pygame.KEYUP:
		# 		if event.key == pygame.K_SPACE:
		# 			paused = not paused
		# if not paused:
		# 	update(next_states_critic)
		# 	render(states_critic, weights_prd)


		states_critic,states_actor = next_states_critic,next_states_actor
		states = next_states
		time.sleep(0.5)


if __name__ == '__main__':
	env = make_env(scenario_name=ENVIRONMENT_NAME,benchmark=False)
	run(env,100)