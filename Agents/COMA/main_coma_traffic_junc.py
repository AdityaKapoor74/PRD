import numpy as np
import os
import gym
import ma_gym
from macoma_traffic_junc import MACOMA


if __name__ == '__main__':

	for i in range(1,6):
		extension = "COMA_run"+str(i)
		test_num = "COMA"
		env_name = "ma_gym:TrafficJunction10-v0"

		dictionary = {
				"critic_dir": "../../../tests/COMA_"+env_name+"/models/critic_networks/run"+str(i)+"/",
				"actor_dir": "../../../tests/COMA_"+env_name+"/models/actor_networks/run"+str(i)+"/",
				"gif_dir": "../../../tests/COMA_"+env_name+"/gif_dir/run"+str(i)+"/",
				"policy_eval_dir":"../../../tests/COMA_"+env_name+"/policy_eval_dir/run"+str(i)+"/",
				"env": env_name, 
				"test_num":test_num,
				"extension":extension,
				"iteration": i,
				"value_lr": 1e-3, 
				"policy_lr": 7e-4,
				"grad_clip_critic": 10.0,
				"grad_clip_actor": 10.0,
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
				"max_episodes": 20000,
				"max_time_steps": 40,
				"norm_adv": False,
				"norm_rew": False,
			}
		env = gym.make(env_name)
		ma_controller = MACOMA(env,dictionary)
		ma_controller.run()
