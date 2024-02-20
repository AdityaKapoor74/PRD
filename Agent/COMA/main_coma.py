from macoma import MACOMA
import gym
import smaclite  # noqa


if __name__ == '__main__':

	RENDER = False
	USE_CPP_RVO2 = False

	for i in range(1,4):
		extension = "COMA_"+str(i)
		test_num = "StarCraft"
		env_name = "3s5z"

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
				"critic_rnn_num_layers": 1, 
				"critic_rnn_hidden_dim": 64,
				"actor_rnn_num_layers": 1,
				"actor_rnn_hidden_dim": 64,
				"update_episode_interval": 10,
				"data_chunk_length": 10,
				"num_updates": 5,
				"enable_grad_clip_critic": True,
				"enable_grad_clip_actor": True,
				"grad_clip_critic": 0.5,
				"grad_clip_actor": 0.5,
				"critic_entropy_pen": 0.0,
				"epsilon_start": 1.0,
				"epsilon_end": 0.05,
				"entropy_pen": 1e-2, #8e-3
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
				"max_episodes": 20000,
				"max_time_steps": 100,
				"norm_adv": False,
				"norm_rew": False,
			}
		env = gym.make(f"smaclite/{env_name}-v0", use_cpp_rvo2=USE_CPP_RVO2)
		obs, info = env.reset(return_info=True)
		dictionary["local_observation"] = obs[0].shape[0]+env.n_agents
		dictionary["global_observation"] = info["ally_states"].shape[1] + info["enemy_states"].reshape(-1).shape[0] + env.n_agents
		ma_controller = MACOMA(env,dictionary)
		ma_controller.run()
