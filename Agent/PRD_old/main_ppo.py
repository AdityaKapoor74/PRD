from mappo import MAPPO
import torch
import gym
import smaclite  # noqa


if __name__ == '__main__':
	RENDER = False
	USE_CPP_RVO2 = False
	
	for i in range(1,4):
		extension = "PRD_old_"+str(i)
		test_num = "StarCraft"
		env_name = "5m_vs_6m"
		experiment_type = "prd_above_threshold_ascend" # shared, prd_above_threshold_ascend, prd_above_threshold, prd_top_k, prd_above_threshold_decay

		torch.autograd.set_detect_anomaly(True)

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
				"update_ppo_agent": 10, # update ppo agent after every update_ppo_agent episodes
				"test_num":test_num,
				"extension":extension,
				"gamma": 0.99,
				"gif": False,
				"gif_checkpoint":1,
				"load_models": False,
				"model_path_value": "../../../tests/PRD_2_MPE/models/crossing_team_greedy_prd_above_threshold_MAPPO_Q_run_2/critic_networks/critic_epsiode100000.pt",
				"model_path_policy": "../../../tests/PRD_2_MPE/models/crossing_team_greedy_prd_above_threshold_MAPPO_Q_run_2/actor_networks/actor_epsiode100000.pt",
				"eval_policy": False,
				"save_model": False,
				"save_model_checkpoint": 1000,
				"save_comet_ml_plot": False,
				"learn":True,
				"max_episodes": 20000,
				"max_time_steps": 100,
				"experiment_type": experiment_type,
				"parallel_training": False,
				"scheduler_need": False,
				"norm_rewards": False,
				"clamp_rewards": False,
				"clamp_rewards_value_min": 0.0,
				"clamp_rewards_value_max": 2.0,


				# ENVIRONMENT
				"env": env_name,

				# CRITIC
				"rnn_num_layers_critic": 1,
				"rnn_hidden_critic_dim": 64,
				"attention_drop_prob": 0.0,
				"value_lr": 5e-4, #1e-3
				"value_weight_decay": 5e-4,
				"enable_grad_clip_critic": False,
				"grad_clip_critic": 10.0,
				"value_clip": 0.2,
				"enable_hard_attention": False,
				"num_heads": 1,
				"critic_weight_entropy_pen": 0.0,
				"critic_score_regularizer": 0.0,
				"target_calc_style": "GAE", # GAE, TD_Lambda, N_steps
				"n_steps": 5,
				"td_lambda": 0.95, # 1 --> Monte Carlo; 0 --> TD(1)
				"norm_returns": False,
				"norm_returns_critic": True,
				

				# ACTOR
				"data_chunk_length": 10,
				"rnn_num_layers_actor": 1,
				"rnn_hidden_actor_dim": 64,
				"enable_grad_clip_actor": False,
				"grad_clip_actor": 10.0,
				"policy_clip": 0.2,
				"policy_lr": 5e-4, #prd 1e-4
				"policy_weight_decay": 5e-4,
				"entropy_pen": 1e-2, #8e-3
				"gae_lambda": 0.95,
				"select_above_threshold": 0.0,
				"threshold_min": 0.0, 
				"threshold_max": 0.2,
				"steps_to_take": 1000,
				"top_k": 0,
				"norm_adv": True,

				"network_update_interval": 1,
			}

		seeds = [42, 142, 242, 342, 442]
		torch.manual_seed(seeds[dictionary["iteration"]-1])
		env = gym.make(f"smaclite/{env_name}-v0", use_cpp_rvo2=USE_CPP_RVO2)
		obs, info = env.reset(return_info=True)
		dictionary["ally_observation"] = info["ally_states"].shape[1]+env.n_agents
		dictionary["enemy_observation"] = info["enemy_states"].shape[1]
		dictionary["local_observation"] = obs[0].shape[0]+env.n_agents
		ma_controller = MAPPO(env,dictionary)
		ma_controller.run()
