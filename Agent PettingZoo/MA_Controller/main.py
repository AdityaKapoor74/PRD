from pettingzoo.magent import battle_v2
from multiagent import MultiAgent


if __name__ == '__main__':
	seed_num = 0 # [0,1,2,3,4]

	extension = "MAPPO_Q" # [MAPPO_Q, MAA2C_Q, MAPPO_Q_Semi_Hard_Attn, MAA2C_Q_Semi_Hard_Attn]
	test_num = "PettingZoo"
	env_name = "Battle"
	experiment_type = "prd_above_threshold" # shared, prd_above_threshold

	dictionary = {
			"iteration": seed_num,
			"update_type": "ppo", # [ppo, a2c]
			"attention_type": "soft", # [semi-hard, soft]
			"grad_clip_critic": 10.0,
			"grad_clip_actor": 10.0,
			"device": "gpu",
			"critic_dir": '../../../tests/'+test_num+'/models/'+env_name+'_'+experiment_type+'_'+extension+'/critic_networks/',
			"actor_dir": '../../../tests/'+test_num+'/models/'+env_name+'_'+experiment_type+'_'+extension+'/actor_networks/',
			"gif_dir": '../../../tests/'+test_num+'/gifs/'+env_name+'_'+experiment_type+'_'+extension+'/',
			"policy_eval_dir":'../../../tests/'+test_num+'/policy_eval/'+env_name+'_'+experiment_type+'_'+extension+'/',
			"policy_clip": 0.05,
			"value_clip": 0.05,
			"n_epochs": 5,
			"update_ppo_agent": 1, # update ppo agent after every 'update_ppo_agent' episodes
			"env": env_name, 
			"test_num":test_num,
			"extension":extension,
			"value_lr": 1e-4,
			"policy_lr": 1e-4,
			"entropy_pen": 1e-3,
			"gamma": 0.99, 
			"gae_lambda": 0.95,
			"lambda": 0.95, # 1 --> Monte Carlo; 0 --> TD(1)
			"select_above_threshold": 0.05, # [0.05: 'soft', 0.0: 'semi-hard' --> attention_type]
			"gif": False,
			"gif_checkpoint":1,
			"load_models": False,
			"model_path_value": "",
			"model_path_policy": "",
			"eval_policy": True,
			"save_model": True,
			"save_model_checkpoint": 1000,
			"save_comet_ml_plot": True,
			"learn":True,
			"max_episodes": 100000,
			"max_time_steps": 75,
			"experiment_type": experiment_type,
		}
	parallel_env = battle_v2.parallel_env(map_size=28, attack_opponent_reward=5)
	parallel_env.seed(1)
	ma_controller = MultiAgent(parallel_env, dictionary)
	ma_controller.run()