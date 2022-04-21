import numpy as np
import gym
import ma_gym
from mappo import MAPPO


if __name__ == '__main__':

	for i in range(1,2):
		extension = "MAPPO_Q_run_"+str(i)
		test_num = "PRD_2_MPE"
		env_name = "ma_gym:Combat-v0"
		experiment_type = "prd_above_threshold" # shared, prd_above_threshold_ascend, prd_above_threshold_decay, prd_above_threshold

		dictionary = {
				"iteration": i,
				"grad_clip_critic": 0.5,
				"grad_clip_actor": 10.0,
				"device": "gpu",
				"update_learning_rate_with_prd": False,
				"critic_dir": '../../../tests/'+test_num+'/models/'+env_name+'_'+experiment_type+'_'+extension+'/critic_networks/',
				"actor_dir": '../../../tests/'+test_num+'/models/'+env_name+'_'+experiment_type+'_'+extension+'/actor_networks/',
				"gif_dir": '../../../tests/'+test_num+'/gifs/'+env_name+'_'+experiment_type+'_'+extension+'/',
				"policy_eval_dir":'../../../tests/'+test_num+'/policy_eval/'+env_name+'_'+experiment_type+'_'+extension+'/',
				"policy_clip": 0.05,
				"value_clip": 0.05,
				"n_epochs": 5,
				"update_ppo_agent": 1, # update ppo agent after every update_ppo_agent episodes
				"env": env_name, 
				"test_num":test_num,
				"extension":extension,
				"value_lr": 5e-4, #1e-3
				"policy_lr": 5e-4, #prd 1e-4
				"entropy_pen": 8e-3, #8e-3
				"critic_weight_entropy_pen": 0.0,
				"gamma": 0.99, 
				"gae_lambda": 0.95,
				"lambda": 0.95, # 1 --> Monte Carlo; 0 --> TD(1)
				"select_above_threshold": 0.1,
				"threshold_min": 0.0, 
				"threshold_max": 0.05,
				"steps_to_take": 1000,
				"top_k": 0,
				"gif": False,
				"gif_checkpoint":1,
				"load_models": False,
				"model_path_value": "../../../tests/PRD_2_MPE/models/crossing_team_greedy_shared_MAPPO_Q_run_1/critic_networks/critic_epsiode11000.pt",
				"model_path_policy": "../../../tests/PRD_2_MPE/models/crossing_team_greedy_shared_MAPPO_Q_run_1/actor_networks/actor_epsiode11000.pt",
				"eval_policy": False,
				"save_model": False,
				"save_model_checkpoint": 1000,
				"save_comet_ml_plot": True,
				"learn":True,
				"max_episodes": 1000000,
				"max_time_steps": 40,
				"experiment_type": experiment_type,
				"norm_adv": False,
				"norm_returns": False,
				"value_normalization": False,
				"parallel_training": False,
			}
		env = gym.make(env_name)
		ma_controller = MAPPO(env,dictionary)
		ma_controller.run()