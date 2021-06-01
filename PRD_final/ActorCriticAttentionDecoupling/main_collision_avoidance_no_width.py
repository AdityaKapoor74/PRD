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

	# experiment_type = "greedy_policy"
	# experiment_type = "top"
	# experiment_type = "without_prd"
	# experiment_type = "with_prd_soft_adv_scaled"
	experiment_type = "with_prd_soft_adv"

	dictionary = {
			"critic_dir": '../../../models/Scalar_dot_product/collision_avoidance_no_width/8_Agents/SingleAttentionMechanism/'+experiment_type+'/critic_networks/',
			"actor_dir": '../../../models/Scalar_dot_product/collision_avoidance_no_width/8_Agents/SingleAttentionMechanism/'+experiment_type+'/actor_networks/',
			"tensorboard_dir":'../../../runs/Scalar_dot_product/collision_avoidance_no_width/8_Agents/SingleAttentionMechanism/'+experiment_type+'/',
			"gif_dir": '../../../gifs/Scalar_dot_product/collision_avoidance_no_width/8_Agents/SingleAttentionMechanism/'+experiment_type+'/',
			"env": "collision_avoidance_no_width", 
			"experiment_type":experiment_type,
			# "experiment_type":"greedy_policy",
			# "experiment_type":"top",
			# "experiment_type":"with_prd_soft_adv",
			# "experiment_type":"with_prd_soft_adv_scaled",
			# "experiment_type":"greedy_and_top",
			"value_lr": 1e-2, #1e-2 for single head
			"policy_lr": 1e-3, # 2e-4 for single head
			"entropy_pen": 0.008,
			"l1_pen":0.1, 
			"gamma": 0.99,
			"trace_decay": .98,
			"select_above_threshold": 0.1,
			"softmax_cut_threshold": 0.1,
			"top_k": 2,
			"gif": False,
			"save": True,
			"max_episodes": 100000,
			"max_time_steps": 100,
			"critic_version":5
		}
	env = make_env(scenario_name=dictionary["env"],benchmark=False)
	ma_controller = MAA2C(env,dictionary)
	ma_controller.run()