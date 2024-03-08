from mappo import MAPPO
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
	
	for i in range(1,4):
		extension = "MAPPO_"+str(i)
		test_num = "StarCraft"
		environment = "PettingZoo" # StarCraft/ MPE/ PressurePlate/ PettingZoo/ LBForaging
		if "LBForaging" in environment:
			num_players = 6
			num_food = 9
			grid_size = 12
			fully_coop = False
		env_name = "pursuit_v4" # 5m_vs_6m/ 10m_vs_11m/ 3s5z/ crossing_team_greedy/ pressureplate-linear-6p-v0/ pursuit_v4/ "Foraging-{0}x{0}-{1}p-{2}f{3}-v2".format(grid_size, num_players, num_food, "-coop" if fully_coop else "")
		experiment_type = "prd_above_threshold_ascend" # shared, prd_above_threshold_ascend, prd_above_threshold, prd_top_k, prd_above_threshold_decay, prd_soft_advantage, prd_soft_advantage_global

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
				"update_ppo_agent": 5, # update ppo agent after every update_ppo_agent episodes
				"environment": environment,
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
				"max_episodes": 5000,
				"max_time_steps": 500,
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
				"enable_grad_clip_critic": True,
				"grad_clip_critic": 10.0,
				"value_clip": 0.2,
				"enable_hard_attention": False,
				"num_heads": 1,
				"critic_weight_entropy_pen": 0.0,
				"critic_score_regularizer": 0.0,
				"target_calc_style": "GAE", # GAE, TD_Lambda, N_steps
				"n_steps": 5,
				"td_lambda": 0.95, # 1 --> Monte Carlo; 0 --> TD(1)
				"norm_returns": True,
				

				# ACTOR
				"data_chunk_length": 10,
				"rnn_num_layers_actor": 1,
				"rnn_hidden_actor_dim": 64,
				"enable_grad_clip_actor": True,
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
		
		if "StarCraft" in environment:
			
			import gym
			import smaclite  # noqa
			
			env = gym.make(f"smaclite/{env_name}-v0", use_cpp_rvo2=USE_CPP_RVO2)
			obs, info = env.reset(return_info=True)
			dictionary["ally_observation"] = info["ally_states"].shape[1]+env.n_agents
			dictionary["enemy_observation"] = info["enemy_states"].shape[1]+env.n_enemies
			dictionary["local_observation"] = obs[0].shape[0]+env.n_agents+env.action_space[0].n
			dictionary["num_agents"] = env.n_agents
			dictionary["num_actions"] = env.action_space[0].n

		elif "MPE" in environment:

			from multiagent.environment import MultiAgentEnv
			import multiagent.scenarios as scenarios

			env, actor_observation_shape, critic_observation_shape = make_env(scenario_name=dictionary["env"], benchmark=False)
			dictionary["ally_observation"] = critic_observation_shape
			dictionary["local_observation"] = actor_observation_shape+env.action_space[0].n
			dictionary["num_agents"] = env.n
			dictionary["num_actions"] = env.action_space[0].n

		elif "PressurePlate" in environment:

			import pressureplate
			import gym

			env = gym.make(env_name)
			dictionary["ally_observation"] = 133+env.n_agents
			dictionary["local_observation"] = 133+5+env.n_agents
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
			
			dictionary["ally_observation"] = obs_range*obs_range*3 + num_agents
			dictionary["local_observation"] = obs_range*obs_range*3 + num_agents + num_actions
			dictionary["num_agents"] = num_agents
			dictionary["num_actions"] = num_actions

		elif "LBForaging" in environment:

			import lbforaging
			import gym

			env = gym.make(env_name, max_episode_steps=dictionary["max_time_steps"], penalty=0.0, normalize_reward=True)
			dictionary["ally_observation"] = num_players*3 + num_food*3 + num_players
			dictionary["local_observation"] = num_players*3 + num_food*3 + num_players + env.action_space[0].n
			dictionary["num_agents"] = num_players
			dictionary["num_actions"] = env.action_space[0].n


		ma_controller = MAPPO(env,dictionary)
		ma_controller.run()
