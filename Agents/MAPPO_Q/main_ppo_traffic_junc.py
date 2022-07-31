import numpy as np
import os
import gym
import ma_gym
from mappo_traffic_junc import MAPPO


if __name__ == '__main__':

	for i in range(1,6):
		extension = "MAA2C_Q_run_"+str(i)
		test_num = "traffic_junction"
		env_name = "ma_gym:TrafficJunction10-v0"
		experiment_type = "prd_above_threshold" # shared, prd_above_threshold_ascend, prd_above_threshold_decay, prd_above_threshold

		dictionary = {
				"iteration": i,
				"update_type": "a2c",
				"grad_clip_critic": 10.0,
				"grad_clip_actor": 10.0,
				"device": "gpu",
				"update_learning_rate_with_prd": False,
				"critic_dir": '../../../tests/'+test_num+'/models/'+env_name+'_'+experiment_type+'_'+extension+'/critic_networks/',
				"actor_dir": '../../../tests/'+test_num+'/models/'+env_name+'_'+experiment_type+'_'+extension+'/actor_networks/',
				"gif_dir": '../../../tests/'+test_num+'/gifs/'+env_name+'_'+experiment_type+'_'+extension+'/',
				"policy_eval_dir":'../../../tests/'+test_num+'/policy_eval/'+env_name+'_'+experiment_type+'_'+extension+'/',
				"policy_clip": 0.2,
				"value_clip": 0.2,
				"n_epochs": 10,
				"update_ppo_agent": 2, # update ppo agent after every update_ppo_agent episodes
				"env": env_name, 
				"test_num":test_num,
				"extension":extension,
				"value_lr": 1e-4, #1e-3
				"policy_lr": 1e-4, #prd 1e-4
				"entropy_pen": 0.0, #8e-3
				"critic_weight_entropy_pen": 0.0,
				"gamma": 0.99, 
				"gae_lambda": 0.95,
				"lambda": 0.95, # 1 --> Monte Carlo; 0 --> TD(1)
				"select_above_threshold": 0.1,
				"threshold_min": 0.0, 
				"threshold_max": 0.0,
				"steps_to_take": 1000,
				"top_k": 0,
				"gif": False,
				"gif_checkpoint":1,
				"load_models": False,
				"model_path_value": "../../../tests/PRD_2_MPE/models/crossing_team_greedy_shared_MAPPO_Q_run_1/critic_networks/critic_epsiode11000.pt",
				"model_path_policy": "../../../tests/PRD_2_MPE/models/crossing_team_greedy_shared_MAPPO_Q_run_1/actor_networks/actor_epsiode11000.pt",
				"eval_policy": True,
				"save_model": True,
				"save_model_checkpoint": 10,
				"save_comet_ml_plot": True,
				"learn":True,
				"max_episodes": 20000,
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


	# GRAD VAR EXPERIMENT
	# episode_list = [str(1000*(i+1)) for i in range(20)]
	# # episode_list = [str(4000)]

	# grad_vars = []
	# grad_stds = []

	# try: 
	# 	os.makedirs("../../../tests/Traffic_Junc/grad_var/", exist_ok = True) 
	# 	print("Eval Directory created successfully") 
	# except OSError as error: 
	# 	print("Eval Directory can not be created")

	# for episode in episode_list:
	# 	extension = "MAA2C_Q_run"
	# 	test_num = "traffic_junction"
	# 	env_name = "ma_gym:TrafficJunction10-v0"
	# 	experiment_type = "shared" # shared, prd_above_threshold_ascend, prd_above_threshold_decay, prd_above_threshold

	# 	dictionary = {
	# 			"iteration": 0,
	# 			"update_type": "ppo",
	# 			"grad_clip_critic": 10.0,
	# 			"grad_clip_actor": 10.0,
	# 			"device": "gpu",
	# 			"update_learning_rate_with_prd": False,
	# 			"critic_dir": '../../../tests/'+test_num+'/models/'+env_name+'_'+experiment_type+'_'+extension+'/critic_networks/',
	# 			"actor_dir": '../../../tests/'+test_num+'/models/'+env_name+'_'+experiment_type+'_'+extension+'/actor_networks/',
	# 			"gif_dir": '../../../tests/'+test_num+'/gifs/'+env_name+'_'+experiment_type+'_'+extension+'/',
	# 			"policy_eval_dir":'../../../tests/'+test_num+'/policy_eval/'+env_name+'_'+experiment_type+'_'+extension+'/',
	# 			"policy_clip": 0.2,
	# 			"value_clip": 0.2,
	# 			"n_epochs": 10,
	# 			"update_ppo_agent": 2, # update ppo agent after every update_ppo_agent episodes
	# 			"env": env_name, 
	# 			"test_num":test_num,
	# 			"extension":extension,
	# 			"value_lr": 5e-5, #1e-3
	# 			"policy_lr": 5e-5, #prd 1e-4
	# 			"entropy_pen": 0.0, #8e-3
	# 			"critic_weight_entropy_pen": 0.0,
	# 			"gamma": 0.99, 
	# 			"gae_lambda": 0.95,
	# 			"lambda": 0.95, # 1 --> Monte Carlo; 0 --> TD(1)
	# 			"select_above_threshold": 0.1,
	# 			"threshold_min": 0.0, 
	# 			"threshold_max": 0.0,
	# 			"steps_to_take": 1000,
	# 			"top_k": 0,
	# 			"gif": False,
	# 			"gif_checkpoint":1,
	# 			"load_models": False,
	# 			"model_path_value": "../../../tests/Traffic_Junc/models/ma_gym:TrafficJunction10-v0_prd_above_threshold_MAPPO_Q_run_1/critic_networks/critic_epsiode"+episode+".pt",
	# 			"model_path_policy": "../../../tests/Traffic_Junc/models/ma_gym:TrafficJunction10-v0_prd_above_threshold_MAPPO_Q_run_1/actor_networks/actor_epsiode"+episode+".pt",
	# 			"eval_policy": False,
	# 			"save_model": False,
	# 			"save_model_checkpoint": 10,
	# 			"save_comet_ml_plot": False,
	# 			"learn":False,
	# 			"max_episodes": 100,
	# 			"max_time_steps": 40,
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
	# 	np.save("../../../tests/Traffic_Junc/grad_var/grad_vars_"+experiment_type+"_"+env_name,np.array(grad_vars))
	# 	np.save("../../../tests/Traffic_Junc/grad_var/grad_stds_"+experiment_type+"_"+env_name,np.array(grad_vars))
	# 	print("grad_vars", grad_vars)
	# 	print("grad_stds", grad_stds)