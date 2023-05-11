from macoma import MACOMA
import lbforaging
import gym
import torch


if __name__ == '__main__':

	for i in range(1,6):
		extension = "COMA_run"+str(i)
		test_num = "LB-FORAGING"
		num_players = 6
		num_food = 9
		grid_size = 12
		fully_coop = False
		max_episode_steps = 70
		env_name = "Foraging-{0}x{0}-{1}p-{2}f{3}-v2".format(grid_size, num_players, num_food, "-coop" if fully_coop else "")

		dictionary = {
				"critic_dir": "../../../tests/COMA_"+env_name+"/models/critic_networks/run"+str(i)+"/",
				"actor_dir": "../../../tests/COMA_"+env_name+"/models/actor_networks/run"+str(i)+"/",
				"gif_dir": "../../../tests/COMA_"+env_name+"/gif_dir/run"+str(i)+"/",
				"policy_eval_dir":"../../../tests/COMA_"+env_name+"/policy_eval_dir/run"+str(i)+"/",
				"env": env_name, 
				"test_num":test_num,
				"extension":extension,
				"iteration": i,
				"value_lr": 1e-4, 
				"policy_lr": 1e-4,
				"grad_clip_critic": 10.0,
				"grad_clip_actor": 10.0,
				"critic_entropy_pen": 0.0,
				"epsilon_start": 1.0,
				"epsilon_end": 0.05,
				"epsilon_episode_steps": 1000,
				"entropy_pen": 1e-3,
				"target_critic_update": 200,
				"gamma": 0.99,
				"lambda": 0.7,
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
				"max_episodes": 30000,
				"max_time_steps": max_episode_steps,
				"norm_adv": False,
				"norm_rew": False,
			}
		seeds = [42, 142, 242, 342, 442]
		torch.manual_seed(seeds[dictionary["iteration"]-1])
		env = gym.make(env_name, max_episode_steps=max_episode_steps, penalty=0.01, normalize_reward=True)
		dictionary["global_observation"] = num_players*3 + num_food*3
		dictionary["local_observation"] = num_players*3 + num_food*3
		ma_controller = MACOMA(env,dictionary)
		ma_controller.run()
