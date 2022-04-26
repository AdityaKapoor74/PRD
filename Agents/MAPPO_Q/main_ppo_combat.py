import numpy as np
import gym
import ma_gym
from mappo_combat import MAPPO_COMBAT


if __name__ == '__main__':

	for i in range(1,6):
		extension = "MAPPO_Q_run_"+str(i)
		test_num = "Combat_PRD_vs_Shared"
		env_name = "ma_gym:Combat-v0"
		experiment_type = "prd_vs_shared" # shared, prd_above_threshold_ascend, prd_above_threshold_decay, prd_above_threshold

		dictionary = {
				"iteration": i,
				"device": "gpu",
				"critic_dir": '../../../tests/'+test_num+'/models/'+env_name+'_'+experiment_type+'_'+extension+'/critic_networks/',
				"actor_dir": '../../../tests/'+test_num+'/models/'+env_name+'_'+experiment_type+'_'+extension+'/actor_networks/',
				"gif_dir": '../../../tests/'+test_num+'/gifs/'+env_name+'_'+experiment_type+'_'+extension+'/',
				"policy_eval_dir":'../../../tests/'+test_num+'/policy_eval/'+env_name+'_'+experiment_type+'_'+extension+'/',
				"env": env_name, 
				"test_num":test_num,
				"extension":extension,
				"gamma": 0.99, 
				"gae_lambda": 0.95,
				"lambda_": 0.95, # 1 --> Monte Carlo; 0 --> TD(1)
				"n_epochs": 5,
				"update_ppo_agents": 1, # update ppo agent after every update_ppo_agent episodes
				"gif": False,
				"gif_checkpoint":1,
				"load_models": False,
				"eval_policy": True,
				"save_model": True,
				"save_model_checkpoint": 1000,
				"save_comet_ml_plot": True,
				"learn":True,
				"max_episodes": 100000,
				"max_time_steps": 40,
				"experiment_type": experiment_type,
				"norm_adv": False,
				"norm_returns": False,
				"value_normalization": False,
				"parallel_training": False,

				# PRD
				"prd_type": "prd_above_threshold",
				"update_learning_rate_with_prd": False,
				"prd_grad_clip_critic": 10.0,
				"prd_grad_clip_actor": 10.0,
				"prd_policy_clip": 0.05,
				"prd_value_clip": 0.05,
				"prd_value_lr": 1e-3, #1e-3
				"prd_policy_lr": 1e-3, #prd 1e-4
				"prd_entropy_pen": 1e-3, #8e-3
				"prd_critic_weight_entropy_pen": 0.0,
				"select_above_threshold": 0.2,
				"threshold_min": 0.0, 
				"threshold_max": 0.0,
				"steps_to_take": 1000,
				"top_k": 0,
				"prd_model_path_value": "../../../tests/PRD_2_MPE/models/crossing_team_greedy_shared_MAPPO_Q_run_1/critic_networks/critic_epsiode11000.pt",
				"prd_model_path_policy": "../../../tests/PRD_2_MPE/models/crossing_team_greedy_shared_MAPPO_Q_run_1/actor_networks/actor_epsiode11000.pt",
				
				# SHARED
				"shared_grad_clip_critic": 10.0,
				"shared_grad_clip_actor": 10.0,
				"shared_policy_clip": 0.05,
				"shared_value_clip": 0.05,
				"shared_value_lr": 1e-3, #1e-3
				"shared_policy_lr": 1e-3, #prd 1e-4
				"shared_entropy_pen": 1e-3, #8e-3
				"shared_critic_weight_entropy_pen": 0.0,
				"shared_model_path_value": "../../../tests/PRD_2_MPE/models/crossing_team_greedy_shared_MAPPO_Q_run_1/critic_networks/critic_epsiode11000.pt",
				"shared_model_path_policy": "../../../tests/PRD_2_MPE/models/crossing_team_greedy_shared_MAPPO_Q_run_1/actor_networks/actor_epsiode11000.pt",
			}
		env = gym.make(env_name)
		ma_controller = MAPPO_COMBAT(env,dictionary)
		ma_controller.run()