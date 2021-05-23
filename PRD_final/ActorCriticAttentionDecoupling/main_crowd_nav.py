# from maa2c import MAA2C
from maa2c_crowd_nav import MAA2C

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
			"critic_dir": '../../../models/Scalar_dot_product/crowd_nav/6_Agents_2_People/SingleAttentionMechanism/with_prd_soft_adv/critic_networks/',
			"actor_dir": '../../../models/Scalar_dot_product/crowd_nav/6_Agents_2_People/SingleAttentionMechanism/with_prd_soft_adv/actor_networks/',
			"tensorboard_dir":'../../../runs/Scalar_dot_product/crowd_nav/6_Agents_2_People/SingleAttentionMechanism/with_prd_soft_adv/',
			"gif_dir": '../../../gifs/Scalar_dot_product/crowd_nav/6_Agents_2_People/SingleAttentionMechanism/with_prd_soft_adv/',
			"env": "crowd_nav", 
			"experiment_type": "with_prd_soft_adv",
			"num_agents": 6,
			"num_people": 2,
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
			"max_episodes": 80000,
			"max_time_steps": 100,
		}
	env = make_env(scenario_name=dictionary["env"],benchmark=False)
	ma_controller = MAA2C(env,dictionary)
	ma_controller.run()