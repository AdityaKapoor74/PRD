# from maa2c import MAA2C
from maa2c_collision_avoidance import MAA2C

from multiagent.environment import MultiAgentEnv
# from multiagent.scenarios.simple_spread import Scenario
import multiagent.scenarios as scenarios
import torch 
import numpy as np

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


def run_file(dictionary):
	env = make_env(scenario_name=dictionary["env"],benchmark=False)
	ma_controller = MAA2C(env,dictionary)
	ma_controller.run()



if __name__ == '__main__':
	dictionary = {
			"critic_dir": '../../../collision_avoidance_no_collision_pen_16_Agents/models/without_prd/critic_networks/',
			"actor_dir": '../../../collision_avoidance_no_collision_pen_16_Agents/models/without_prd/actor_networks/',
			"tensorboard_dir":'../../../collision_avoidance_no_collision_pen_16_Agents/runs/without_prd/',
			"gif_dir": '../../../collision_avoidance_no_collision_pen_16_Agents/gifs/without_prd/',
			"env": "collision_avoidance", 
			"value_lr": 1e-2, #1e-2 for single head
			"policy_lr": 5e-4, # 2e-4 for single head
			"entropy_pen": 0.008, 
			"gamma": 0.99,
			"trace_decay": 0.98,
			"select_above_threshold": 0.1,
			"softmax_cut_threshold": 0.1,
			"experiment_type": "without_prd",
			"top_k": 0,
			"gif": False,
			"save": True,
			"max_episodes": 100000,
			"max_time_steps": 100,
		}
	env = make_env(scenario_name=dictionary["env"],benchmark=False)
	ma_controller = MAA2C(env,dictionary)
	ma_controller.run()