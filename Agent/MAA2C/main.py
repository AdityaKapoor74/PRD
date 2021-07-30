from maa2c import MAA2C

from multiagent.environment import MultiAgentEnv
import multiagent.scenarios as scenarios

def make_env(scenario_name, benchmark=False):
	scenario = scenarios.load(scenario_name + ".py").Scenario()
	world = scenario.make_world()
	if benchmark:
		env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation, scenario.benchmark_data, scenario.isFinished)
	else:
		env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation, None, scenario.isFinished)
	return env


def run_file(dictionary):
	env = make_env(scenario_name=dictionary["env"],benchmark=False)
	ma_controller = MAA2C(env,dictionary)
	ma_controller.run()


if __name__ == '__main__':

	# color_social_dilemma_pt2
	for i in range(1,6):
		extension = "run"+str(i)
		test_num = "color_social_dilemma_pt2_8_Agents"
		env_name = "color_social_dilemma_pt2" 
		experiment_type = "greedy" # prd_above_threshold_decay_episodic, greedy, shared, prd_above_threshold_ascend_episodic

		dictionary = {
				"critic_dir": '../../../tests/'+test_num+'/models/'+env_name+'_'+experiment_type+'_'+extension+'/critic_networks/',
				"actor_dir": '../../../tests/'+test_num+'/models/'+env_name+'_'+experiment_type+'_'+extension+'/actor_networks/',
				"tensorboard_dir":'../../../tests/'+test_num+'/runs/'+env_name+'_'+experiment_type+'_'+extension+'/',
				"gif_dir": '../../../tests/'+test_num+'/gifs/'+env_name+'_'+experiment_type+'_'+extension+'/',
				"env": env_name, 
				"value_lr": 1e-3, 
				"policy_lr": 1e-4,
				"entropy_pen": 0.0, 
				"critic_loss_type": "MC",
				"gamma": 0.99, 
				"trace_decay": 0.98,
				"lambda": 0.0, #0.8
				"select_above_threshold": 0.01,
				"threshold_min": 0.0, #0.0001
				"threshold_max": 0.001,
				"steps_to_take": 10000, #1000
				"top_k": 0,
				"gif": False,
				"save_model": True,
				"save_model_checkpoint": 100,
				"save_tensorboard_plot": True,
				"learn":True,
				"max_episodes": 10000,
				"max_time_steps": 100,
				"experiment_type": experiment_type,
				"gif_checkpoint":100,
				"gae": True,
				"norm_adv": False,
				"norm_rew": False,
			}
		env = make_env(scenario_name=dictionary["env"],benchmark=False)
		ma_controller = MAA2C(env,dictionary)
		ma_controller.run()


	# crossing /  paired_by_sharing_goals
	# for i in range(1,6):
	# 	extension = "run_MAA2C_MC_prd_above_threshold_decay_episodic"+str(i)
	# 	test_num = "crossing"
	# 	env_name = "crossing"
	# 	experiment_type = "prd_above_threshold_decay_episodic" # prd_above_threshold_decay_episodic, greedy, shared

	# 	dictionary = {
	# 			"critic_dir": '../../../tests/'+test_num+'/models/'+env_name+'_'+experiment_type+'_'+extension+'/critic_networks/',
	# 			"actor_dir": '../../../tests/'+test_num+'/models/'+env_name+'_'+experiment_type+'_'+extension+'/actor_networks/',
	# 			"tensorboard_dir":'../../../tests/'+test_num+'/runs/'+env_name+'_'+experiment_type+'_'+extension+'/',
	# 			"gif_dir": '../../../tests/'+test_num+'/gifs/'+env_name+'_'+experiment_type+'_'+extension+'/',
	# 			"env": env_name, 
	# 			"value_lr": 1e-3, 
	# 			"policy_lr": 1e-4,
	# 			"entropy_pen": 0.0, 
	# 			"critic_loss_type": "MC",
	# 			"gamma": 0.99, 
	# 			"trace_decay": 0.98,
	# 			"lambda": 0.0, #0.8
	# 			"select_above_threshold": 0.01,
	# 			"threshold_min": 0.0, 
	# 			"steps_to_take": 100000, 
	# 			"top_k": 0,
	# 			"gif": False,
	# 			"save_model": True,
	# 			"save_model_checkpoint": 1000,
	# 			"save_tensorboard_plot": True,
	# 			"learn":True,
	# 			"max_episodes": 100000,
	# 			"max_time_steps": 100,
	# 			"experiment_type": experiment_type,
	# 			"gif_checkpoint":1,
	# 			"gae": True,
	# 			"norm_adv": False,
	# 			"norm_rew": False,
	# 		}
	# 	env = make_env(scenario_name=dictionary["env"],benchmark=False)
	# 	ma_controller = MAA2C(env,dictionary)
	# 	ma_controller.run()


