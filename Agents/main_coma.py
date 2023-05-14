from macoma import MACOMA
import gfootball.env as football_env
import torch


if __name__ == '__main__':

	for i in range(1,6):
		extension = "COMA_run"+str(i)
		test_num = "GOOGLE FOOTBALL"
		env_name = "academy_counterattack_easy"

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
				"policy_lr": 5e-4,
				"grad_clip_critic": 0.5,
				"grad_clip_actor": 0.5,
				"critic_entropy_pen": 0.0,
				"epsilon_start": 1.0,
				"epsilon_end": 0.05,
				"epsilon_episode_steps": 5000,
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
				"max_episodes": 30000,
				"max_time_steps": 40,
				"norm_adv": False,
				"norm_rew": False,
				"num_agents": 6,
			}
		seeds = [42, 142, 242, 342, 442]
		torch.manual_seed(seeds[dictionary["iteration"]-1])
		env = football_env.create_environment(
			env_name=env_name,
			number_of_left_players_agent_controls=dictionary["num_agents"],
			# number_of_right_players_agent_controls=2,
			representation="simple115",
			# num_agents=4,
			stacked=False, 
			logdir='/tmp/football', 
			write_goal_dumps=False, 
			write_full_episode_dumps=False, 
			rewards='scoring,checkpoints',
			render=False
			)
		dictionary["global_observation"] = 115
		dictionary["local_observation"] = 115
		ma_controller = MACOMA(env,dictionary)
		ma_controller.run()
