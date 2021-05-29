# from maa2c import MAA2C
from maa2c_coma import MAA2C

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
		env = MultiAgentEnv(world, scenario.reset_world, scenario.reward_paired_agents, scenario.observation, scenario.benchmark_data, scenario.isFinished)
	else:
		env = MultiAgentEnv(world, scenario.reset_world, scenario.reward_paired_agents, scenario.observation, None, scenario.isFinished)
	return env


def run_file(dictionary):
	env = make_env(scenario_name=dictionary["env"],benchmark=False)
	ma_controller = MAA2C(env,dictionary)
	ma_controller.run()



if __name__ == '__main__':
	dictionary = {
			"critic_dir": '../../../paired_agents_4_Agents_coma/models/critic_networks/',
			"actor_dir": '../../../paired_agents_4_Agents_coma/models/actor_networks/',
			"tensorboard_dir":'../../../paired_agents_4_Agents_coma/runs/',
			"gif_dir": '../../../paired_agents_4_Agents_coma/gifs/',
			"env": "paired_by_sharing_goals", 
			"experiment_type":"coma",
			"value_lr": 1e-2, #1e-2 for single head
			"policy_lr": 1e-3, # 2e-4 for single head
			"entropy_pen": 0.008, 
			"gamma": 0.99,
			"trace_decay": 0.98,
			"select_above_threshold": 0.1,
			"softmax_cut_threshold": 0.1,
			"top_k": 2,
			"gif": False,
			"save": True,
			"max_episodes": 200000,
			"max_time_steps": 100,
		}
	env = make_env(scenario_name=dictionary["env"],benchmark=False)
	ma_controller = MAA2C(env,dictionary)
	ma_controller.run()