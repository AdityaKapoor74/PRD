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

	# VERSION 1
	# dictionary = {
	# 		"version": 1,
	# 		"nstep": True,
	# 		"critic_dir": '../../../paired_agents_4_Agents_coma_v1/models/critic_networks/',
	# 		"actor_dir": '../../../paired_agents_4_Agents_coma_v1/models/actor_networks/',
	# 		"tensorboard_dir":'../../../paired_agents_4_Agents_coma_v1/runs/',
	# 		"gif_dir": '../../../paired_agents_4_Agents_coma_v1/gifs/',
	# 		"env": "paired_by_sharing_goals", 
	# 		"experiment_type":"coma_v1",
	# 		"value_lr": 1e-2,
	# 		"policy_lr": 1e-4,
	# 		"entropy_pen": 0.0008, 
	# 		"gamma": 0.99,
	# 		"trace_decay": 0.98,
	# 		"select_above_threshold": 0.1,
	# 		"softmax_cut_threshold": 0.1,
	# 		"top_k": 0,
	# 		"gif": False,
	# 		"save": True,
	# 		"max_episodes": 100000,
	# 		"max_time_steps": 100,
	# 	}

	# VERSION 2
	dictionary = {
			"version": 2,
			"nstep": True,
			"critic_dir": '../../../paired_agents_4_Agents_coma_v2/models/critic_networks/',
			"actor_dir": '../../../paired_agents_4_Agents_coma_v2/models/actor_networks/',
			"tensorboard_dir":'../../../paired_agents_4_Agents_coma_v2/runs/',
			"gif_dir": '../../../paired_agents_4_Agents_coma_v2/gifs/',
			"env": "paired_by_sharing_goals", 
			"experiment_type":"coma_v2",
			"value_lr": 1e-2,
			"policy_lr": 9e-4, 
			"entropy_pen": 0.001, 
			"gamma": 0.99,
			"trace_decay": 0.98,
			"select_above_threshold": 0.1,
			"softmax_cut_threshold": 0.1,
			"top_k": 0,
			"gif": False,
			"save": True,
			"max_episodes": 100000,
			"max_time_steps": 100,
		}

	# VERSION 3
	# dictionary = {
	# 		"version": 3,
	# 		"nstep": True,
	# 		"critic_dir": '../../../paired_agents_4_Agents_coma_v3/models/critic_networks/',
	# 		"actor_dir": '../../../paired_agents_4_Agents_coma_v3/models/actor_networks/',
	# 		"tensorboard_dir":'../../../paired_agents_4_Agents_coma_v3/runs/',
	# 		"gif_dir": '../../../paired_agents_4_Agents_coma_v3/gifs/',
	# 		"env": "paired_by_sharing_goals", 
	# 		"experiment_type":"coma_v3",
	# 		"value_lr": 1e-2,
	# 		"policy_lr": 75e-4, 
	# 		"entropy_pen": 0.008, 
	# 		"gamma": 0.99,
	# 		"trace_decay": 0.98,
	# 		"select_above_threshold": 0.1,
	# 		"softmax_cut_threshold": 0.1,
	# 		"top_k": 0,
	# 		"gif": False,
	# 		"save": True,
	# 		"max_episodes": 100000,
	# 		"max_time_steps": 100,
	# 	}

	# VERSION 4
	# dictionary = {
	# 		"version": 4,
	# 		"nstep": True,
	# 		"critic_dir": '../../../paired_agents_4_Agents_coma_v4/models/critic_networks/',
	# 		"actor_dir": '../../../paired_agents_4_Agents_coma_v4/models/actor_networks/',
	# 		"tensorboard_dir":'../../../paired_agents_4_Agents_coma_v4/runs/',
	# 		"gif_dir": '../../../paired_agents_4_Agents_coma_v4/gifs/',
	# 		"env": "paired_by_sharing_goals", 
	# 		"experiment_type":"coma_v4",
	# 		"value_lr": 1e-2,
	# 		"policy_lr": 1e-3, 
	# 		"entropy_pen": 0.008, 
	# 		"gamma": 0.99,
	# 		"trace_decay": 0.98,
	# 		"select_above_threshold": 0.1,
	# 		"softmax_cut_threshold": 0.1,
	# 		"top_k": 0,
	# 		"gif": False,
	# 		"save": True,
	# 		"max_episodes": 100000,
	# 		"max_time_steps": 100,
	# 	}

	# VERSION 5
	# dictionary = {
	# 		"version": 5,
	# 		"nstep": True,
	# 		"critic_dir": '../../../paired_agents_4_Agents_coma_v5_nstep/models/critic_networks/',
	# 		"actor_dir": '../../../paired_agents_4_Agents_coma_v5_nstep/models/actor_networks/',
	# 		"tensorboard_dir":'../../../paired_agents_4_Agents_coma_v5_nstep/runs/',
	# 		"gif_dir": '../../../paired_agents_4_Agents_coma_v5_nstep/gifs/',
	# 		"env": "paired_by_sharing_goals", 
	# 		"experiment_type":"coma_v5",
	# 		"value_lr": 1e-2,
	# 		"policy_lr": 1e-3, 
	# 		"entropy_pen": 0.008, 
	# 		"gamma": 0.99,
	# 		"trace_decay": 0.98,
	# 		"select_above_threshold": 0.1,
	# 		"softmax_cut_threshold": 0.1,
	# 		"top_k": 0,
	# 		"gif": False,
	# 		"save": True,
	# 		"max_episodes": 100000,
	# 		"max_time_steps": 100,
	# 	}

	# VERSION 6
	# dictionary = {
	# 		"version": 6,
	# 		"nstep": True,
	# 		"critic_dir": '../../../paired_agents_4_Agents_coma_v6_n_step_return/models/critic_networks/',
	# 		"actor_dir": '../../../paired_agents_4_Agents_coma_v6_n_step_return/models/actor_networks/',
	# 		"tensorboard_dir":'../../../paired_agents_4_Agents_coma_v6_n_step_return/runs/',
	# 		"gif_dir": '../../../paired_agents_4_Agents_coma_v6_n_step_return/gifs/',
	# 		"env": "paired_by_sharing_goals", 
	# 		"experiment_type":"coma_v6",
	# 		"value_lr": 1e-2,
	# 		"policy_lr": 1e-3, 
	# 		"entropy_pen": 0.008, 
	# 		"gamma": 0.99,
	# 		"trace_decay": 0.98,
	# 		"select_above_threshold": 0.1,
	# 		"softmax_cut_threshold": 0.1,
	# 		"top_k": 0,
	# 		"gif": False,
	# 		"save": True,
	# 		"max_episodes": 100000,
	# 		"max_time_steps": 100,
	# 	}

	env = make_env(scenario_name=dictionary["env"],benchmark=False)
	ma_controller = MAA2C(env,dictionary)
	ma_controller.run()