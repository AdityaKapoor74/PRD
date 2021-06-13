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
		env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation, scenario.benchmark_data, scenario.isFinished)
	else:
		env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation, None, scenario.isFinished)
	return env


def run_file(dictionary):
	env = make_env(scenario_name=dictionary["env"],benchmark=False)
	ma_controller = MAA2C(env,dictionary)
	ma_controller.run()



if __name__ == '__main__':

	# VERSION 1
	# for i in range(5):
	# 	dictionary = {
	# 			"version": 1,
	# 			"nstep": True,
				# "critic_dir": '../../../paired_agents_4_Agents_coma_v1/models'+str(i)+'/critic_networks/',
				# "actor_dir": '../../../paired_agents_4_Agents_coma_v1/models'+str(i)+'/actor_networks/',
				# "policy_eval_dir": "../../../paired_agents_4_Agents_coma_v1/policy_eval"+str(i)+"/",
				# "tensorboard_dir":'../../../paired_agents_4_Agents_coma_v1/runs'+str(i)+'/',
	# 			"gif_dir": '../../../paired_agents_4_Agents_coma_v1/gifs'+str(i)+'/',
	# 			"env": "paired_by_sharing_goals", 
	# 			"experiment_type":"coma_v1",
	# 			"value_lr": 1e-2,
	# 			"policy_lr": 1e-4,
	# 			"entropy_pen": 8e-4, 
	# 			"gamma": 0.99,
	# 			"trace_decay": 0.98,
	# 			"select_above_threshold": 0.1,
	# 			"softmax_cut_threshold": 0.1,
	# 			"top_k": 0,
	# 			"gif": False,
	# 			"save": True,
	# 			"max_episodes": 60000,
	# 			"max_time_steps": 100,
	# 		}

		# env = make_env(scenario_name=dictionary["env"],benchmark=False)
		# ma_controller = MAA2C(env,dictionary)
		# ma_controller.run()

	# VERSION 2
	for i in range(5):
		dictionary = {
				"version": 2,
				"nstep": True,
				"critic_dir": '../../../paired_agents_4_Agents_coma_v2/models'+str(i)+'/critic_networks/',
				"actor_dir": '../../../paired_agents_4_Agents_coma_v2/models'+str(i)+'/actor_networks/',
				"policy_eval_dir": "../../../paired_agents_4_Agents_coma_v2/policy_eval"+str(i)+"/",
				"tensorboard_dir":'../../../paired_agents_4_Agents_coma_v2/runs'+str(i)+'/',
				"gif_dir": '../../../paired_agents_4_Agents_coma_v2/gifs'+str(i)+'/',
				"env": "paired_by_sharing_goals", 
				"experiment_type":"coma_v2",
				"value_lr": 1e-2,
				"policy_lr": 5e-4, 
				"entropy_pen": 0.0, 
				"gamma": 0.99,
				"trace_decay": 0.98,
				"select_above_threshold": 0.1,
				"softmax_cut_threshold": 0.1,
				"top_k": 0,
				"gif": False,
				"save": True,
				"max_episodes": 60000,
				"max_time_steps": 100,
			}

		env = make_env(scenario_name=dictionary["env"],benchmark=False)
		ma_controller = MAA2C(env,dictionary)
		ma_controller.run()

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
	# 		"policy_lr": 5e-4, 
	# 		"entropy_pen": 1e-1, 
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
	# for i in range(5):
	# 	dictionary = {
	# 			"version": 4,
	# 			"nstep": True,
	# 			"critic_dir": '../../../paired_agents_4_Agents_coma_v4/models'+str(i)+'/critic_networks/',
	# 			"actor_dir": '../../../paired_agents_4_Agents_coma_v4/models'+str(i)+'/actor_networks/',
	# 			"policy_eval_dir": "../../../paired_agents_4_Agents_coma_v4/policy_eval"+str(i)+"/",
	# 			"tensorboard_dir":'../../../paired_agents_4_Agents_coma_v4/runs'+str(i)+'/',
	# 			"gif_dir": '../../../paired_agents_4_Agents_coma_v4/gifs'+str(i)+'/',
	# 			"env": "paired_by_sharing_goals", 
	# 			"experiment_type":"coma_v4",
	# 			"value_lr": 1e-2,
	# 			"policy_lr": 1e-3, 
	# 			"entropy_pen": 0.008, 
	# 			"gamma": 0.99,
	# 			"trace_decay": 0.98,
	# 			"select_above_threshold": 0.1,
	# 			"softmax_cut_threshold": 0.1,
	# 			"top_k": 0,
	# 			"gif": False,
	# 			"save": True,
	# 			"max_episodes": 60000,
	# 			"max_time_steps": 100,
	# 		}
	# 	env = make_env(scenario_name=dictionary["env"],benchmark=False)
	# 	ma_controller = MAA2C(env,dictionary)
	# 	ma_controller.run()

	# VERSION 5
	# for i in range(5):
	# 	dictionary = {
	# 			"version": 5,
	# 			"nstep": True,
	# 			"critic_dir": '../../../paired_agents_4_Agents_coma_v5/models'+str(i)+'/critic_networks/',
	# 			"actor_dir": '../../../paired_agents_4_Agents_coma_v5/models'+str(i)+'/actor_networks/',
	# 			"policy_eval_dir": "../../../paired_agents_4_Agents_coma_v5/policy_eval"+str(i)+"/",
	# 			"tensorboard_dir":'../../../paired_agents_4_Agents_coma_v5/runs'+str(i)+'/',
	# 			"gif_dir": '../../../paired_agents_4_Agents_coma_v5/gifs'+str(i)+'/',
	# 			"env": "paired_by_sharing_goals", 
	# 			"experiment_type":"coma_v5",
	# 			"value_lr": 1e-2,
	# 			"policy_lr": 1e-3, 
	# 			"entropy_pen": 0.008, 
	# 			"gamma": 0.99,
	# 			"trace_decay": 0.98,
	# 			"select_above_threshold": 0.1,
	# 			"softmax_cut_threshold": 0.1,
	# 			"top_k": 0,
	# 			"gif": False,
	# 			"save": True,
	# 			"max_episodes": 100000,
	# 			"max_time_steps": 100,
	# 		}
	# 	env = make_env(scenario_name=dictionary["env"],benchmark=False)
	# 	ma_controller = MAA2C(env,dictionary)
	# 	ma_controller.run()


	# VERSION 6
	# for i in range(5):
	# 	dictionary = {
	# 			"version": 6,
	# 			"nstep": True,
	# 			"critic_dir": '../../../paired_agents_4_Agents_coma_v6/models'+str(i)+'/critic_networks/',
	# 			"actor_dir": '../../../paired_agents_4_Agents_coma_v6/models'+str(i)+'/actor_networks/',
	# 			"policy_eval_dir": "../../../paired_agents_4_Agents_coma_v6/policy_eval"+str(i)+"/",
	# 			"tensorboard_dir":'../../../paired_agents_4_Agents_coma_v6/runs'+str(i)+'/',
	# 			"gif_dir": '../../../paired_agents_4_Agents_coma_v6/gifs'+str(i)+'/',
	# 			"env": "paired_by_sharing_goals", 
	# 			"experiment_type":"coma_v6",
	# 			"value_lr": 1e-2,
	# 			"policy_lr": 1e-3, 
	# 			"entropy_pen": 0.008, 
	# 			"gamma": 0.99,
	# 			"trace_decay": 0.98,
	# 			"select_above_threshold": 0.1,
	# 			"softmax_cut_threshold": 0.1,
	# 			"top_k": 0,
	# 			"gif": False,
	# 			"save": True,
	# 			"max_episodes": 60000,
	# 			"max_time_steps": 100,
	# 		}

	# 	env = make_env(scenario_name=dictionary["env"],benchmark=False)
	# 	ma_controller = MAA2C(env,dictionary)
	# 	ma_controller.run()

	# env = make_env(scenario_name=dictionary["env"],benchmark=False)
	# ma_controller = MAA2C(env,dictionary)
	# ma_controller.run()