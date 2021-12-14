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
	# crossing_greedy/ crossing_fully_coop /  paired_by_sharing_goals/ crossing_partially_coop/ color_social_dilemma
	for i in range(1,6):
		extension = "MAA2C_run_"+str(i)
		test_num = "PRD_2_exps" 
		env_name = "crossing_greedy"
		experiment_type = "prd_above_threshold_ascend" # prd_above_threshold_ascend, greedy, shared

		dictionary = {
				"policy_type": "MLP", # MLP/ Transformer/ DualTransformer/ MultiHeadTransformer/ MultiHeadDualTransformer
				"critic_type": "Transformer", # Transformer/ DualTransformer/ MultiHeadTransformer/ MultiHeadDualTransformer
				"num_heads_critic": 1,
				"num_heads_actor": 1,
				"critic_dir": '../../../data/MAA2C_'+env_name+"_"+experiment_type+'/models/critic_networks/run'+str(i),
				"actor_dir": '../../../data/MAA2C_'+env_name+"_"+experiment_type+'/models/actor_networks/run'+str(i),
				"gif_dir": '../../../data/MAA2C_'+env_name+"_"+experiment_type+'/gifs/',
				"policy_eval_dir":'../../../data/MAA2C_'+env_name+"_"+experiment_type+'/policy_eval_dir/run'+str(i),
				"env": env_name, 
				"test_num":test_num,
				"extension":extension,
				"iteration": i,
				"device": "cpu",
				"value_lr": 7e-4, #1e-3 
				"policy_lr": 5e-4, #prd 1e-4
				"entropy_pen": 8e-3, #8e-3
				"entropy_pen_min": 1e-3, #8e-3
				"tau": 1e-3,
				"target_critic_update_eps": 200,
				"l1_pen": 0.0,
				"critic_entropy_pen": 0.0,
				"use_target_net": False,
				"critic_loss_type": "TD_lambda",
				"target_critic_update": "hard",
				"gamma": 0.99, 
				"trace_decay": 0.98,
				"lambda": 0.8, #0.8
				"select_above_threshold": 0.0,
				"threshold_min": 0.0, 
				"threshold_max": 0.05,
				"steps_to_take": 1000, 
				"l1_pen_min": 0.0,
				"l1_pen_steps_to_take": 0,
				"top_k": 0,
				"gif": False,
				"gif_checkpoint":1,
				"load_models": True,
				"model_path_value": " ",
				"model_path_policy": " ",
				"eval_policy": False,
				"save_model": False,
				"save_model_checkpoint": 1000,
				"save_comet_ml_plot": False,
				"learn":True,
				"max_episodes": 50000,
				"max_time_steps": 100,
				"experiment_type": experiment_type,
				"gae": True,
				"norm_adv": False,
				"norm_rew": False,
			}
		env = make_env(scenario_name=dictionary["env"],benchmark=False)
		ma_controller = MAA2C(env,dictionary)
		ma_controller.run()
