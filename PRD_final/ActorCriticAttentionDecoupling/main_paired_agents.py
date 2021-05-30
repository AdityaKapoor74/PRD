from maa2c_paired_agents import MAA2C

from multiagent.environment import MultiAgentEnv
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
			"critic_dir": '../../../models/Scalar_dot_product/paired_by_sharing_goals/4_Agents/SingleAttentionMechanism/with_prd_gnn_policy/critic_networks/',
			"actor_dir": '../../../models/Scalar_dot_product/paired_by_sharing_goals/4_Agents/SingleAttentionMechanism/with_prd_gnn_policy/actor_networks/',
			"tensorboard_dir":'../../../runs/Scalar_dot_product/paired_by_sharing_goals/4_Agents/SingleAttentionMechanism/with_prd_gnn_policy/',
			"gif_dir": '../../../gifs/Scalar_dot_product/paired_by_sharing_goals/4_Agents/SingleAttentionMechanism/with_prd_gnn_policy/',
			"env": "paired_by_sharing_goals",
			# "env":"collision_avoidance", 
			"value_lr": 1e-2, #1e-2 for single head
			"policy_lr": 1e-3, # 2e-4 for single head
			"entropy_pen": 0.008, 
			"gamma": 0.99,
			"trace_decay": .98,
			"select_above_threshold": 0.1,
			"softmax_cut_threshold": 0.1,
			"top_k": 2,
			"gif": False,
			"save": True,
			"max_episodes": 80000,
			"max_time_steps": 100,
			#"experiment_type": "without_prd",
			# "experiment_type":"with_prd_soft_adv",
			"experiment_type":"greedy_policy",
			# "experiment_type":"with_prd_soft_adv_scaled"
			"critic_version":5,
			"obs_act_output_dim":64
		}
	env = make_env(scenario_name=dictionary["env"],benchmark=False)
	ma_controller = MAA2C(env,dictionary)
	ma_controller.run()