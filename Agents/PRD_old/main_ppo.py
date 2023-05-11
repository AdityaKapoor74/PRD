from mappo import MAPPO
import lbforaging
import gym
import torch

'''
crossing_team_greedy
PRD_MAA2C: value_lr = 5e-4; policy_lr = 3e-4; entropy_pen = 8e-3; grad_clip_critic = 0.5; grad_clip_actor = 0.5; threshold = 0.05

crossing_greedy
PRD_MAA2C_Q: value_lr = 5e-4; policy_lr = 3e-4; entropy_pen = 8e-3; grad_clip_critic = 0.5; grad_clip_actor = 0.5; threshold = 0.1
'''

if __name__ == '__main__':
	# crossing_greedy/ crossing_fully_coop /  paired_by_sharing_goals/ crossing_partially_coop/ color_social_dilemma
	for i in range(1,6):
		extension = "PRD_old_"+str(i)
		test_num = "LB-FORAGING"
		num_players = 6
		num_food = 9
		grid_size = 12
		fully_coop = False
		max_episode_steps = 70
		env_name = "Foraging-{0}x{0}-{1}p-{2}f{3}-v2".format(grid_size, num_players, num_food, "-coop" if fully_coop else "")
		experiment_type = "prd_above_threshold_ascend" # prd_above_threshold_ascend, greedy, shared

		dictionary = {
				# TRAINING
				"iteration": i,
				"device": "gpu",
				"update_learning_rate_with_prd": False,
				"critic_dir": '../../../tests/'+test_num+'/models/'+env_name+'_'+experiment_type+'_'+extension+'/critic_networks/',
				"actor_dir": '../../../tests/'+test_num+'/models/'+env_name+'_'+experiment_type+'_'+extension+'/actor_networks/',
				"gif_dir": '../../../tests/'+test_num+'/gifs/'+env_name+'_'+experiment_type+'_'+extension+'/',
				"policy_eval_dir":'../../../tests/'+test_num+'/policy_eval/'+env_name+'_'+experiment_type+'_'+extension+'/',
				"n_epochs": 10,
				"update_ppo_agent": 7, # update ppo agent after every update_ppo_agent episodes
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
				"max_time_steps": max_episode_steps,
				"experiment_type": experiment_type,
				"parallel_training": False,
				"scheduler_need": False,


				# ENVIRONMENT
				"env": env_name,

				# CRITIC
				"value_lr": 1e-4, #1e-3
				"value_weight_decay": 5e-4,
				"grad_clip_critic": 0.5,
				"value_clip": 0.2,
				"enable_hard_attention": False,
				"num_heads": 4,
				"critic_weight_entropy_pen": 0.0,
				"critic_score_regularizer": 0.0,
				"lambda": 0.95, # 1 --> Monte Carlo; 0 --> TD(1)
				"norm_returns": False,
				

				# ACTOR
				"grad_clip_actor": 0.5,
				"policy_clip": 0.2,
				"policy_lr": 1e-4, #prd 1e-4
				"policy_weight_decay": 5e-4,
				"entropy_pen": 1e-2, #8e-3
				"gae_lambda": 0.95,
				"select_above_threshold": 0.0,
				"threshold_min": 0.0, 
				"threshold_max": 0.01,
				"steps_to_take": 1000,
				"top_k": 0,
				"norm_adv": False,

				"network_update_interval": 1,
			}

		seeds = [42, 142, 242, 342, 442]
		torch.manual_seed(seeds[dictionary["iteration"]-1])
		env = gym.make(env_name, max_episode_steps=max_episode_steps, penalty=0.01, normalize_reward=True)
		dictionary["global_observation"] = num_players*3 + num_food*3
		dictionary["local_observation"] = num_players*3 + num_food*3
		ma_controller = MAPPO(env,dictionary)
		ma_controller.run()
