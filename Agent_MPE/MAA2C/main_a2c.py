from maa2c import MAA2C

from multiagent.environment import MultiAgentEnv
import multiagent.scenarios as scenarios

import time
import torch

def make_env(scenario_name, benchmark=False):
	scenario = scenarios.load(scenario_name + ".py").Scenario()
	world = scenario.make_world()
	if benchmark:
		env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation, scenario.benchmark_data, scenario.isFinished)
	else:
		env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation, None, scenario.isFinished)
	return env, scenario.observation_shape, scenario.transformer_observation_shape


def run_file(dictionary):
	env = make_env(scenario_name=dictionary["env"],benchmark=False)
	ma_controller = MAA2C(env,dictionary)
	ma_controller.run()

'''
crossing_team_greedy
PRD_MAA2C: value_lr = 5e-4; policy_lr = 3e-4; entropy_pen = 8e-3; grad_clip_critic = 0.5; grad_clip_actor = 0.5; threshold = 0.05

crossing_greedy
PRD_MAA2C_Q: value_lr = 5e-4; policy_lr = 3e-4; entropy_pen = 8e-3; grad_clip_critic = 0.5; grad_clip_actor = 0.5; threshold = 0.1
'''

if __name__ == '__main__':
	# crossing_greedy/ crossing_fully_coop /  paired_by_sharing_goals/ crossing_partially_coop/ color_social_dilemma
	for i in range(1,6):
		extension = "MAA2C_run_"+str(i)
		test_num = "TEAM COLLISION AVOIDANCE" 
		env_name = "crossing_team_greedy"
		experiment_type = "prd_above_threshold_ascend" # prd_above_threshold_ascend, greedy, shared

		dictionary = {
				"critic_dir": '../../../tests/'+test_num+'/models/'+env_name+'_'+experiment_type+'_'+extension+'/critic_networks/',
				"actor_dir": '../../../tests/'+test_num+'/models/'+env_name+'_'+experiment_type+'_'+extension+'/actor_networks/',
				"gif_dir": '../../../tests/'+test_num+'/gifs/'+env_name+'_'+experiment_type+'_'+extension+'/',
				"policy_eval_dir":'../../../tests/'+test_num+'/policy_eval/'+env_name+'_'+experiment_type+'_'+extension+'/',
				"env": env_name, 
				"test_num":test_num,
				"extension":extension,
				"iteration": i,
				"device": "gpu",
				"value_lr": 1e-4, #1e-3 
				"policy_lr": 1e-4, #prd 1e-4
				"grad_clip_critic": 0.5,
				"grad_clip_actor": 0.5,
				"entropy_pen": 8e-3, #8e-3
				"entropy_pen_min": 0.0, #8e-3
				"critic_entropy_pen": 0.0,
				"critic_loss_type": "TD_lambda",
				"gamma": 0.99, 
				"trace_decay": 0.95,
				"lambda": 0.95, #0.8
				"select_above_threshold": 0.0,
				"threshold_min": 0.0, 
				"threshold_max": 0.043,
				"steps_to_take": 1000, 
				"l1_pen": 0.0,
				"l1_pen_min": 0.0,
				"l1_pen_steps_to_take": 0,
				"top_k": 0,
				"update_after_episodes": 1,
				"gif": False,
				"gif_checkpoint":1,
				"load_models": False,
				"model_path_value": " ",
				"model_path_policy": " ",
				"eval_policy": True,
				"save_model": True,
				"save_model_checkpoint": 1000,
				"save_comet_ml_plot": True,
				"learn":True,
				"max_episodes": 50000,
				"max_time_steps": 100,
				"experiment_type": experiment_type,
				"gae": True,
				"norm_adv": False,
				"norm_rew": False,
			}
		seeds = [42, 142, 242, 342, 442]
		torch.manual_seed(seeds[dictionary["iteration"]-1])
		env, local_observation, global_observation = make_env(scenario_name=dictionary["env"],benchmark=False)
		dictionary["global_observation"] = global_observation
		dictionary["local_observation"] = local_observation
		ma_controller = MAA2C(env,dictionary)
		ma_controller.run()
