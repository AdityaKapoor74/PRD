import gym
import numpy as np
import os
import pressureplate
from mappo import MAPPO

'''
PRD_MAA2C_Q: value_lr = 5e-4; policy_lr = 3e-4; entropy_pen = 8e-3; grad_clip_critic = 0.5; grad_clip_actor = 0.5; threshold = 0.1
PRD_MAPPO_Q: value_lr = 5e-4; policy_lr = 5e-4; entropy_pen = 0.4; grad_clip_critic = 0.5; grad_clip_actor = 0.5; value_clip = 0.1; policy_clip = 0.1; ppo_epochs = 1; threshold = 0.1; ppo_update_batch_size = 1
'''


if __name__ == '__main__':

	for i in range(1,3):
		extension = "MAPPO_Q_run_"+str(i)
		test_num = "Pressure Plate"
		env_name = "pressureplate-linear-4p-v0" # paired_by_sharing_goals, color_social_dilemma, crossing_team_greedy, crossing_greedy, crossing_partially_coop, crossing_fully_coop
		experiment_type = "prd_above_threshold" # prd_above_threshold_decay, prd_above_threshold_ascend, shared

		dictionary = {
				"iteration": i,
				"update_type": "ppo",
				"grad_clip_critic": 0.5,
				"grad_clip_actor": 0.5,
				"device": "gpu",
				"update_learning_rate_with_prd": False,
				"critic_dir": '../../../tests/'+test_num+'/models/'+env_name+'_'+experiment_type+'_'+extension+'/critic_networks/',
				"actor_dir": '../../../tests/'+test_num+'/models/'+env_name+'_'+experiment_type+'_'+extension+'/actor_networks/',
				"gif_dir": '../../../tests/'+test_num+'/gifs/'+env_name+'_'+experiment_type+'_'+extension+'/',
				"policy_eval_dir":'../../../tests/'+test_num+'/policy_eval/'+env_name+'_'+experiment_type+'_'+extension+'/',
				"policy_clip": 0.1,
				"value_clip": 0.1,
				"n_epochs": 1,
				"update_ppo_agent": 1, # update ppo agent after every update_ppo_agent episodes
				"env": env_name, 
				"test_num":test_num,
				"extension":extension,
				"value_lr": 5e-4, #1e-3
				"policy_lr": 5e-4, #prd 1e-4
				"entropy_pen": 0.4, #8e-3
				"critic_weight_entropy_pen": 0.0,
				"gamma": 0.99, 
				"gae_lambda": 0.95,
				"lambda": 0.95, # 1 --> Monte Carlo; 0 --> TD(1)
				"select_above_threshold": 0.1,
				"threshold_min": 0.0, 
				"threshold_max": 0.05,
				"steps_to_take": 1000,
				"top_k": 0,
				"gif": True,
				"gif_checkpoint":1,
				"load_models": True,
				"model_path_value": "../../../tests/PRD_PRESSURE_PLATE/models/pressureplate-linear-4p-v0_prd_above_threshold_MAPPO_Q_run_3/critic_networks/critic_epsiode20000.pt",
				"model_path_policy": "../../../tests/PRD_PRESSURE_PLATE/models/pressureplate-linear-4p-v0_prd_above_threshold_MAPPO_Q_run_3/actor_networks/actor_epsiode20000.pt",
				"eval_policy": False,
				"save_model": False,
				"save_model_checkpoint": 1000,
				"save_comet_ml_plot": False,
				"learn":True,
				"max_episodes": 20000,
				"max_time_steps": 70,
				"experiment_type": experiment_type,
				"norm_adv": False,
				"norm_returns": False,
				"value_normalization": False,
				"parallel_training": False,
			}
		env = gym.make(env_name)
		ma_controller = MAPPO(env,dictionary)
		ma_controller.run()
	# 	# ma_controller.test()
	


	# GRAD VAR EXPERIMENT
	# episode_list = [str(1000*(i+1)) for i in range(20)]
	# # episode_list = [str(4000)]

	# grad_vars = []
	# grad_stds = []

	# try: 
	# 	os.makedirs("../../../tests/Pressure Plate/grad_var/", exist_ok = True) 
	# 	print("Eval Directory created successfully") 
	# except OSError as error: 
	# 	print("Eval Directory can not be created")

	# for episode in episode_list:
	# 	extension = "MAPPO_Q_run"
	# 	test_num = "Pressure Plate"
	# 	env_name = "pressureplate-linear-4p-v0" # paired_by_sharing_goals, color_social_dilemma, crossing_team_greedy, crossing_greedy, crossing_partially_coop, crossing_fully_coop
	# 	experiment_type = "shared" # prd_above_threshold_decay, prd_above_threshold_ascend, shared

	# 	dictionary = {
	# 			"iteration": 3,
	# 			"update_type": "ppo",
	# 			"grad_clip_critic": 0.5,
	# 			"grad_clip_actor": 0.5,
	# 			"device": "gpu",
	# 			"update_learning_rate_with_prd": False,
	# 			"critic_dir": '../../../tests/'+test_num+'/models/'+env_name+'_'+experiment_type+'_'+extension+'/critic_networks/',
	# 			"actor_dir": '../../../tests/'+test_num+'/models/'+env_name+'_'+experiment_type+'_'+extension+'/actor_networks/',
	# 			"gif_dir": '../../../tests/'+test_num+'/gifs/'+env_name+'_'+experiment_type+'_'+extension+'/',
	# 			"policy_eval_dir":'../../../tests/'+test_num+'/policy_eval/'+env_name+'_'+experiment_type+'_'+extension+'/',
	# 			"policy_clip": 0.05,
	# 			"value_clip": 0.05,
	# 			"n_epochs": 5,
	# 			"update_ppo_agent": 1, # update ppo agent after every update_ppo_agent episodes
	# 			"env": env_name, 
	# 			"test_num":test_num,
	# 			"extension":extension,
	# 			"value_lr": 7e-4, #1e-3
	# 			"policy_lr": 7e-4, #prd 1e-4
	# 			"entropy_pen": 0.4, #8e-3
	# 			"critic_weight_entropy_pen": 0.0,
	# 			"gamma": 0.99, 
	# 			"gae_lambda": 0.95,
	# 			"lambda": 0.95, # 1 --> Monte Carlo; 0 --> TD(1)
	# 			"select_above_threshold": 0.05,
	# 			"threshold_min": 0.0, 
	# 			"threshold_max": 0.05,
	# 			"steps_to_take": 1000,
	# 			"top_k": 0,
	# 			"gif": False,
	# 			"gif_checkpoint":1,
	# 			"load_models": True,
	# 			"model_path_value": "../../../tests/Pressure Plate/models/pressureplate-linear-4p-v0_prd_above_threshold_MAPPO_Q_run_3/critic_networks/critic_epsiode"+episode+".pt",
	# 			"model_path_policy": "../../../tests/Pressure Plate/models/pressureplate-linear-4p-v0_prd_above_threshold_MAPPO_Q_run_3/actor_networks/actor_epsiode"+episode+".pt",
	# 			"eval_policy": False,
	# 			"save_model": False,
	# 			"save_model_checkpoint": 1000,
	# 			"save_comet_ml_plot": False,
	# 			"learn":False,
	# 			"max_episodes": 100,
	# 			"max_time_steps": 70,
	# 			"experiment_type": experiment_type,
	# 			"norm_adv": False,
	# 			"norm_returns": False,
	# 			"value_normalization": False,
	# 			"parallel_training": False,
	# 		}

	# 	env = gym.make(env_name)
	# 	ma_controller = MAPPO(env,dictionary)
	# 	grad_var, std = ma_controller.run_gradvar_exp(int(episode))
	# 	grad_vars.append(grad_var)
	# 	grad_stds.append(std)
	# 	np.save("../../../tests/Pressure Plate/grad_var/grad_vars_"+experiment_type+"_"+env_name,np.array(grad_vars))
	# 	np.save("../../../tests/Pressure Plate/grad_var/grad_stds_"+experiment_type+"_"+env_name,np.array(grad_vars))
	# 	print("grad_vars", grad_vars)
	# 	print("grad_stds", grad_stds)