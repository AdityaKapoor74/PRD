# from maa2c import MAA2C
from maa2c_test import MAA2C

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

critic_type = "GNNTanhRelU"
extension = "GNNTanhRelU" # MLP_CRITIC_STATE, MLP_CRITIC_STATE_ACTION, GNN_CRITIC_STATE, GNN_CRITIC_STATE_ACTION, ALL, ALL_W_POL, NonResVx, ResVx, AttentionCriticV1, MLPToGNN
test_num = "test23"
env_name = "multi_circular"
experiment_type = "without_prd"
if __name__ == '__main__':
	dictionary = {
			"critic_dir": '../../../../tests/'+test_num+'/models/'+env_name+'_'+experiment_type+'_'+extension+'/critic_networks/',
			"actor_dir": '../../../../tests/'+test_num+'/models/'+env_name+'_'+experiment_type+'_'+extension+'/actor_networks/',
			"tensorboard_dir":'../../../../tests/'+test_num+'/runs/'+env_name+'_'+experiment_type+'_'+extension+'/',
			"gif_dir": '../../../../tests/'+test_num+'/gifs/'+env_name+'_'+experiment_type+'_'+extension+'/',
			"env": env_name, #paired_by_sharing_goals, multi_circular
			"value_lr": 1e-2, #1e-2 for single head [1e-2, 1e-2, 5e-2, 5e-2]
			"policy_lr": 5e-4, # 2e-4 for single head
			"entropy_pen": 8e-3, 
			"gamma": 0.99,
			"trace_decay": 0.98,
			"select_above_threshold": 1e-1,
			"softmax_cut_threshold": 1e-1,
			"top_k": 0,
			"gif": False,
			"save": True,
			"learn":True,
			"max_episodes": 100000,
			"max_time_steps": 100,
			"experiment_type": experiment_type,
			"critic_type": critic_type,
			"gif_checkpoint":10,
			"gae": True,
			"norm_adv": True,
			"norm_rew": False,
			"attention_heads": 4,
			"freeze_policy": 100000,
		}
	env = make_env(scenario_name=dictionary["env"],benchmark=False)
	ma_controller = MAA2C(env,dictionary)
	ma_controller.run()


'''
Paired agent --> value_lr = 1e-2 / policy_lr = 1e-3
Multi Circular --> value_lr = 1e-3 / policy_lr = 5e-4
'''