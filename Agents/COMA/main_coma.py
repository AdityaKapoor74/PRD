from macoma import MACOMA
import numpy as np
import gym
import ma_gym


if __name__ == '__main__':

	for i in range(1,6):
		extension = "COMA_run"+str(i)
		test_num = "COMA"
		env_name = "ma_gym:Combat-v0"

		dictionary = {
				"critic_dir": '../../../tests/'+test_num+'/models/'+env_name+'_'+extension+'/critic_networks/',
				"actor_dir": '../../../tests/'+test_num+'/models/'+env_name+'_'+extension+'/actor_networks/',
				"gif_dir": '../../../tests/'+test_num+'/gifs/'+env_name+'_'+'_'+extension+'/',
				"policy_eval_dir":'../../../tests/'+test_num+'/policy_eval/'+env_name+'_'+extension+'/',
				"env": env_name, 
				"test_num":test_num,
				"extension":extension,
				"iteration": i,
				"value_lr": 3e-4, 
				"policy_lr": 3e-4,
				"grad_clip_critic": 0.5,
				"grad_clip_actor": 0.5,
				"critic_entropy_pen": 0.0,
				"epsilon_start": 0.5,
				"epsilon_end": 0.02,
				"epsilon_episode_steps": 750,
				"target_critic_update": 200,
				"gamma": 0.99,
				"lambda": 0.8,
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
				"max_episodes": 120000,
				"max_time_steps": 40,
				"norm_adv": False,
				"norm_rew": False,
			}
		env = gym.make(env_name)
		ma_controller = MACOMA(env,dictionary)
		ma_controller.run()
