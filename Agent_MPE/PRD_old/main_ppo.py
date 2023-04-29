from mappo import MAPPO
from multiagent.environment import MultiAgentEnv
import multiagent.scenarios as scenarios
import torch

'''
crossing_team_greedy
PRD_MAA2C: value_lr = 5e-4; policy_lr = 3e-4; entropy_pen = 8e-3; grad_clip_critic = 0.5; grad_clip_actor = 0.5; threshold = 0.05

crossing_greedy
PRD_MAA2C_Q: value_lr = 5e-4; policy_lr = 3e-4; entropy_pen = 8e-3; grad_clip_critic = 0.5; grad_clip_actor = 0.5; threshold = 0.1
'''


def make_env(scenario_name, benchmark=False):
	scenario = scenarios.load(scenario_name + ".py").Scenario()
	world = scenario.make_world()
	if benchmark:
		env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation, scenario.benchmark_data, scenario.isFinished)
	else:
		env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation, None, scenario.isFinished)
	return env, scenario.observation_shape, scenario.transformer_observation_shape


if __name__ == '__main__':
	# crossing_greedy/ crossing_fully_coop /  paired_by_sharing_goals/ crossing_partially_coop/ color_social_dilemma
	for i in range(1,6):
		extension = "PRD_old_"+str(i)
		test_num = "TEAM COLLISION AVOIDANCE"
		env_name = "crossing_team_greedy"
		experiment_type = "prd_above_threshold_ascend" # shared, prd_above_threshold_ascend, prd_above_threshold, prd_top_k, prd_above_threshold_decay

		dictionary = {
				# TRAINING
				"iteration": i,
				"device": "gpu",
				"update_learning_rate_with_prd": False,
				"critic_dir": '../../../tests/'+test_num+'/models/'+env_name+'_'+experiment_type+'_'+extension+'/critic_networks/',
				"actor_dir": '../../../tests/'+test_num+'/models/'+env_name+'_'+experiment_type+'_'+extension+'/actor_networks/',
				"gif_dir": '../../../tests/'+test_num+'/gifs/'+env_name+'_'+experiment_type+'_'+extension+'/',
				"policy_eval_dir":'../../../tests/'+test_num+'/policy_eval/'+env_name+'_'+experiment_type+'_'+extension+'/',
				"n_epochs": 5,
				"update_ppo_agent": 5, # update ppo agent after every update_ppo_agent episodes
				"test_num":test_num,
				"extension":extension,
				"gamma": 0.99,
				"gif": False,
				"gif_checkpoint":1,
				"load_models": False,
				"model_path_value": "../../../tests/PRD_2_MPE/models/crossing_team_greedy_prd_above_threshold_MAPPO_Q_run_2/critic_networks/critic_epsiode100000.pt",
				"model_path_policy": "../../../tests/PRD_2_MPE/models/crossing_team_greedy_prd_above_threshold_MAPPO_Q_run_2/actor_networks/actor_epsiode100000.pt",
				"eval_policy": True,
				"save_model": True,
				"save_model_checkpoint": 1000,
				"save_comet_ml_plot": True,
				"learn":True,
				"max_episodes": 30000,
				"max_time_steps": 100,
				"experiment_type": experiment_type,
				"parallel_training": False,
				"scheduler_need": False,


				# ENVIRONMENT
				"team_size": 8,
				"env": env_name,

				# CRITIC
				"value_lr": 1e-4, #1e-3
				"value_weight_decay": 5e-4,
				"grad_clip_critic": 10.0,
				"value_clip": 0.05,
				"enable_hard_attention": False,
				"num_heads": 4,
				"critic_weight_entropy_pen": 0.0,
				"critic_score_regularizer": 0.0,
				"lambda": 0.8, # 1 --> Monte Carlo; 0 --> TD(1)
				"norm_returns": False,
				

				# ACTOR
				"grad_clip_actor": 10.0,
				"policy_clip": 0.05,
				"policy_lr": 5e-5, #prd 1e-4
				"policy_weight_decay": 5e-4,
				"entropy_pen": 7e-1, #8e-3
				"gae_lambda": 0.95,
				"select_above_threshold": 0.0,
				"threshold_min": 0.0, 
				"threshold_max": 0.1,
				"steps_to_take": 1000,
				"top_k": 0,
				"norm_adv": False,

				"network_update_interval": 1,
			}

		seeds = [42, 142, 242, 342, 442]
		torch.manual_seed(seeds[dictionary["iteration"]-1])
		env, local_observation, global_observation = make_env(scenario_name=dictionary["env"],benchmark=False)
		dictionary["global_observation"] = global_observation
		dictionary["local_observation"] = local_observation
		ma_controller = MAPPO(env,dictionary)
		ma_controller.run()
