from mappo import MAPPO
import os
import numpy as np
from multiagent.environment import MultiAgentEnv
import multiagent.scenarios as scenarios

def make_env(scenario_name, benchmark=False):
	scenario = scenarios.load(scenario_name + ".py").Scenario()
	world = scenario.make_world()
	if benchmark:
		env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation, scenario.benchmark_data, scenario.isFinished)
	else:
		env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation, None, scenario.isFinished)
	return env


def run_file(dictionary):
	env = make_env(scenario_name=dictionary["env"],benchmark=False)
	ma_controller = MAA2C(env,dictionary)
	ma_controller.run()

'''
crossing_greedy
PRD_MAA2C_Q: value_lr = 5e-4; policy_lr = 3e-4; entropy_pen = 8e-3; grad_clip_critic = 0.5; grad_clip_actor = 0.5; threshold = 0.1
PRD_MAPPO_Q: value_lr = 5e-4; policy_lr = 5e-4; entropy_pen = 0.0; grad_clip_critic = 0.5; grad_clip_actor = 0.5; value_clip = 0.1; policy_clip = 0.1; ppo_epochs = 1; threshold = 0.1; ppo_update_batch_size = 1

crossing_team_greedy
PRD_MAA2C_Q: value_lr = 5e-4; policy_lr = 3e-4; entropy_pen = 8e-3; grad_clip_critic = 0.5; grad_clip_actor = 0.5; threshold = 0.05
PRD_MAPPO_Q: value_lr = 5e-4; policy_lr = 5e-4; entropy_pen = 0.0; grad_clip_critic = 0.5; grad_clip_actor = 0.5; value_clip = 0.1; policy_clip = 0.1; ppo_epochs = 1; threshold = 0.1; ppo_update_batch_size = 1

paired_agent
PRD_MAA2C_Q: value_lr = 5e-4; policy_lr = 3e-4; entropy_pen = 8e-3; grad_clip_critic = 0.5; grad_clip_actor = 0.5; threshold = 0.1
PRD_MAPPO_Q: value_lr = 5e-4; policy_lr = 3e-4; entropy_pen = 8e-3; grad_clip_critic = 0.5; grad_clip_actor = 0.5; value_clip = 0.05; policy_clip = 0.05; ppo_epochs = 5; threshold = 0.1
'''


if __name__ == '__main__':

	for i in range(1,2):
		extension = "MAPPO_Q_run_"+str(i)
		test_num = "MPE"
		env_name = "crossing_team_greedy"
		experiment_type = "prd_soft_adv" # shared, prd_above_threshold, prd_top_k, prd_above_threshold_decay, prd_above_threshold_ascend

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
				"model_path_value": "./critic_epsiode1000.pt", #"../../../tests/PRD_2_MPE/models/crossing_team_greedy_prd_above_threshold_MAPPO_Q_run_2/critic_networks/critic_epsiode100000.pt",
				"model_path_policy": "./actor_epsiode1000.pt",#"../../../tests/PRD_2_MPE/models/crossing_team_greedy_prd_above_threshold_MAPPO_Q_run_2/actor_networks/actor_epsiode100000.pt",
				"eval_policy": True,
				"save_model": True,
				"save_model_checkpoint": 1000,
				"save_comet_ml_plot": True,
				"learn":True,
				"max_episodes": 20000,
				"max_time_steps": 50,
				"experiment_type": experiment_type,
				"norm_adv": False,
				"norm_returns": False,
				"value_normalization": False,
				"parallel_training": False,
			}
		env = make_env(scenario_name=dictionary["env"],benchmark=False)
		ma_controller = MAPPO(env,dictionary)
		ma_controller.run()
		# ma_controller.test()


	# GRAD VAR EXPERIMENT
	# episode_list = [str(4000*(i+1)) for i in range(20)]
	# # episode_list = [str(4000)]

	# grad_vars = []
	# grad_stds = []

	# try: 
	# 	os.makedirs("../../../tests/Crossing/grad_var/", exist_ok = True) 
	# 	print("Eval Directory created successfully") 
	# except OSError as error: 
	# 	print("Eval Directory can not be created")

	# for episode in episode_list:
	# 	extension = "MAPPO_Q_run"
	# 	test_num = "MPE"
	# 	env_name = "crossing_team_greedy"
	# 	experiment_type = "shared" # shared, prd_above_threshold, prd_top_k, prd_above_threshold_decay, prd_above_threshold_ascend

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
	# 			"policy_clip": 0.05,
	# 			"value_clip": 0.05,
	# 			"n_epochs": 5,
	# 			"update_ppo_agent": 1, # update ppo agent after every update_ppo_agent episodes
	# 			"env": env_name, 
	# 			"test_num":test_num,
	# 			"extension":extension,
	# 			"value_lr": 1e-3, #1e-3
	# 			"policy_lr": 7e-4, #prd 1e-4
	# 			"entropy_pen": 0.0, #8e-3
	# 			"critic_weight_entropy_pen": 0.0,
	# 			"gamma": 0.99, 
	# 			"gae_lambda": 0.95,
	# 			"lambda": 0.95, # 1 --> Monte Carlo; 0 --> TD(1)
	# 			"select_above_threshold": 0.05,
	# 			"threshold_min": 0.0, 
	# 			"threshold_max": 0.0,
	# 			"steps_to_take": 1000,
	# 			"top_k": 5,
	# 			"gif": False,
	# 			"gif_checkpoint":1,
	# 			"load_models": True,
	# 			"model_path_value": "../../../tests/Crossing/models/crossing_team_greedy_prd_above_threshold_MAPPO_Q_run_1/critic_networks/critic_epsiode"+episode+".pt",
	# 			"model_path_policy": "../../../tests/Crossing/models/crossing_team_greedy_prd_above_threshold_MAPPO_Q_run_1/actor_networks/actor_epsiode"+episode+".pt",
	# 			"eval_policy": False,
	# 			"save_model": False,
	# 			"save_model_checkpoint": 1000,
	# 			"save_comet_ml_plot": False,
	# 			"learn":True,
	# 			"max_episodes": 100,
	# 			"max_time_steps": 50,
	# 			"experiment_type": experiment_type,
	# 			"norm_adv": False,
	# 			"norm_returns": False,
	# 			"value_normalization": False,
	# 			"parallel_training": False,
	# 		}
	# 	env = make_env(scenario_name=dictionary["env"],benchmark=False)
	# 	ma_controller = MAPPO(env,dictionary)
	# 	grad_var, std = ma_controller.run_gradvar_exp(int(episode))
	# 	grad_vars.append(grad_var)
	# 	grad_stds.append(std)
	# 	np.save("../../../tests/Crossing/grad_var/grad_vars_"+experiment_type+"_"+env_name,np.array(grad_vars))
	# 	np.save("../../../tests/Crossing/grad_var/grad_stds_"+experiment_type+"_"+env_name,np.array(grad_vars))
	# 	print("grad_vars", grad_vars)
	# 	print("grad_stds", grad_stds)