from macoma import MACOMA
import torch


def make_env(scenario_name, benchmark=False):
	scenario = scenarios.load(scenario_name + ".py").Scenario()
	world = scenario.make_world()
	if benchmark:
		env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation, scenario.benchmark_data, scenario.isFinished)
	else:
		env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation, None, scenario.isFinished)
	return env, scenario.actor_observation_shape, scenario.critic_observation_shape




if __name__ == '__main__':

	RENDER = False
	USE_CPP_RVO2 = False

	torch.set_printoptions(profile="full")
	torch.autograd.set_detect_anomaly(True)

	for i in range(1, 4):
		extension = "COMA_"+str(i)
		test_num = "StarCraft"
		environment = "PettingZoo" # StarCraft/ MPE/ PressurePlate/ PettingZoo/ LBForaging
		if "LBForaging" in environment:
			num_players = 6
			num_food = 9
			grid_size = 12
			fully_coop = False
		env_name = "pursuit_v4" # 5m_vs_6m/ 10m_vs_11m/ 3s5z/ crossing_team_greedy/ pressureplate-linear-6p-v0/ pursuit_v4/ "Foraging-{0}x{0}-{1}p-{2}f{3}-v2".format(grid_size, num_players, num_food, "-coop" if fully_coop else "")

		dictionary = {
				"critic_dir": '../../../tests/'+test_num+'/models/'+env_name+'_'+extension+'/critic_networks/',
				"actor_dir": '../../../tests/'+test_num+'/models/'+env_name+'_'+extension+'/actor_networks/',
				"gif_dir": '../../../tests/'+test_num+'/gifs/'+env_name+'_'+extension+'/',
				"policy_eval_dir":'../../../tests/'+test_num+'/policy_eval/'+env_name+'_'+extension+'/',
				"env": env_name, 
				"test_num":test_num,
				"extension":extension,
				"environment": environment,
				"iteration": i,
				"value_lr": 5e-4, 
				"policy_lr": 5e-4,
				"critic_rnn_num_layers": 1, 
				"critic_rnn_hidden_dim": 64,
				"actor_rnn_num_layers": 1,
				"actor_rnn_hidden_dim": 64,
				"update_episode_interval": 5,
				"data_chunk_length": 10,
				"num_updates": 1,
				"enable_grad_clip_critic": True,
				"enable_grad_clip_actor": True,
				"grad_clip_critic": 10.0,
				"grad_clip_actor": 10.0,
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
				"max_episodes": 5000,
				"max_time_steps": 500,
				"norm_adv": False,
				"norm_rew": False,
			}
		

		if "StarCraft" in environment:
			
			import gym
			import smaclite  # noqa
			
			env = gym.make(f"smaclite/{env_name}-v0", use_cpp_rvo2=USE_CPP_RVO2)
			obs, info = env.reset(return_info=True)
			dictionary["global_observation"] = info["ally_states"].shape[1] + info["enemy_states"].reshape(-1).shape[0] + env.n_agents
			dictionary["local_observation"] = obs[0].shape[0]+env.n_agents
			dictionary["num_agents"] = env.n_agents
			dictionary["num_actions"] = env.action_space[0].n

		elif "MPE" in environment:

			from multiagent.environment import MultiAgentEnv
			import multiagent.scenarios as scenarios

			env, actor_observation_shape, critic_observation_shape = make_env(scenario_name=dictionary["env"], benchmark=False)
			dictionary["global_observation"] = critic_observation_shape
			dictionary["local_observation"] = actor_observation_shape
			dictionary["num_agents"] = env.n
			dictionary["num_actions"] = env.action_space[0].n

		elif "PressurePlate" in environment:

			import pressureplate
			import gym

			env = gym.make(env_name)
			dictionary["global_observation"] = 133+env.n_agents
			dictionary["local_observation"] = 133+env.n_agents
			dictionary["num_agents"] = env.n_agents
			dictionary["num_actions"] = 5

		elif "PettingZoo" in environment:

			from pettingzoo.sisl import pursuit_v4

			num_agents = 8
			num_actions = 5
			obs_range = 7

			env = pursuit_v4.parallel_env(max_cycles=dictionary["max_time_steps"], x_size=16, y_size=16, shared_reward=False, n_evaders=30,
									n_pursuers=num_agents, obs_range=obs_range, n_catch=2, freeze_evaders=False, tag_reward=0.01,
									catch_reward=5.0, urgency_reward=-0.1, surround=True, constraint_window=1.0)
			
			dictionary["global_observation"] = obs_range*obs_range*3 + num_agents
			dictionary["local_observation"] = obs_range*obs_range*3 + num_agents
			dictionary["num_agents"] = num_agents
			dictionary["num_actions"] = num_actions

		elif "LBForaging" in environment:

			import lbforaging
			import gym

			env = gym.make(env_name, max_episode_steps=dictionary["max_time_steps"], penalty=0.0, normalize_reward=True)
			dictionary["global_observation"] = num_players*3 + num_food*3 + num_players
			dictionary["local_observation"] = num_players*3 + num_food*3 + num_players
			dictionary["num_agents"] = num_players
			dictionary["num_actions"] = env.action_space[0].n


		ma_controller = MACOMA(env,dictionary)
		ma_controller.run()
