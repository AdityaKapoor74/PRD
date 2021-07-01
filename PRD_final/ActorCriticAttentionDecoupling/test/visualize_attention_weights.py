import pygame
import sys
import random
from a2c_test import *
from multiagent.environment import MultiAgentEnv
# from multiagent.scenarios.simple_spread import Scenario
import multiagent.scenarios as scenarios
import torch
from torch.distributions import Categorical
import numpy as np

critic_type = "MLPToGNNV6"
num_actions = 5
num_agents = 4
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ENVIRONMENT = "PAIRED AGENTS"
SCREEN_SIZE = WIDTH, HEIGHT = (800, 800)
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 50, 50)
GREEN = (50, 255, 50)
AGENT_RADIUS = 800/20
LANDMARK_RADIUS = 800/40

# MLP_CRITIC_STATE, MLP_CRITIC_STATE_ACTION, GNN_CRITIC_STATE, GNN_CRITIC_STATE_ACTION
if critic_type == "GNN_CRITIC_STATE":
	critic_network = StateOnlyGATCritic(2*3, 128, 128, 1, num_agents, num_actions).to(device)
elif critic_type == "GNN_CRITIC_STATE_ACTION":
	critic_network = StateActionGATCritic(2*3, 128, 2*3+num_actions, 128, 128, 1, num_agents, num_actions).to(device)
elif critic_type == "MLPToGNNV6":
	critic_network = MLPToGNNV6(2*3, 128, 2*3+num_actions, 128, 128, 1, num_agents, num_actions).to(device)

# MLP POLICY
policy_network = MLPPolicyNetwork(2*3, num_agents, num_actions).to(device)


# Loading models
model_path_value = "../../../../tests/test26/models/multi_circular_with_prd_soft_adv_DualMLPGATCritic_MLPTrain_no_adv_norm/critic_networks/29-06-2021VN_ATN_FCN_lr[0.01, 0.01, 0.01, 0.01, 0.01, 0.01]_PN_ATN_FCN_lr0.0005_GradNorm0.5_Entropy0.008_trace_decay0.98topK_0select_above_threshold0.1softmax_cut_threshold0.1_epsiode99000_DualMLPGATCritic_MLPTrain_.pt"
model_path_policy = "../../../../tests/test26/models/multi_circular_with_prd_soft_adv_DualMLPGATCritic_MLPTrain_no_adv_norm/actor_networks/29-06-2021_PN_ATN_FCN_lr0.0005VN_SAT_FCN_lr[0.01, 0.01, 0.01, 0.01, 0.01, 0.01]_GradNorm0.5_Entropy0.008_trace_decay0.98topK_0select_above_threshold0.1softmax_cut_threshold0.1_epsiode99000_DualMLPGATCritic_MLPTrain.pt"
# For CPU
# critic_network.load_state_dict(torch.load(model_path_value,map_location=torch.device('cpu')))
# policy_network.load_state_dict(torch.load(model_path_policy,map_location=torch.device('cpu')))
# # For GPU
critic_network.load_state_dict(torch.load(model_path_value))
policy_network.load_state_dict(torch.load(model_path_policy))

# Initialization
pygame.init()
screen = pygame.display.set_mode(SCREEN_SIZE)
pygame.display.set_caption(ENVIRONMENT)
fps = pygame.time.Clock()
paused = False

agent_position = []
landmark_position = []
agent_color = [random.sample(range(0, 255), 3) for i in range(num_agents)]
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
		posex, posey = scale_pose_to_screen(states[i][4], states[i][5])
		landmark_position.append([posex,posey])


def map_weight_to_width(weight):
	# divide into 10 buckets, weight --> (0,1)
	return round(weight*10)

def update(actions):
	for i in range(num_agents):
		if actions[i] == no_act:
			continue
		elif actions[i] == left_act:
			agent_position[i][0] -= (WIDTH/100)
		elif actions[i] == right_act:
			agent_position[i][0] += (WIDTH/100)
		elif actions[i] == down_act:
			agent_position[i][1] -= (HEIGHT/100)
		elif actions[i] == up_act:
			agent_position[i][1] += (HEIGHT/100)


def render(states_critic, weights):
	screen.fill(BLACK)

	for i in range(num_agents):
		pygame.draw.circle(screen, agent_color[i], agent_position[i], AGENT_RADIUS, 0)
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
			actions = [Categorical(dist).sample().cpu().detach().item() for dist in dists[0]]

			one_hot_actions = np.zeros((num_agents,num_actions))
			for i,act in enumerate(actions):
				one_hot_actions[i][act] = 1
			_, weights = critic_network.forward(torch.FloatTensor([states_critic]).to(device), dists, torch.FloatTensor([one_hot_actions]).to(device))
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
			update(actions)
			render(states_critic, weights[0])


		states_critic,states_actor = next_states_critic,next_states_actor
		states = next_states


if __name__ == '__main__':
	env = make_env(scenario_name="multi_circular",benchmark=False)
	run(env,100000)