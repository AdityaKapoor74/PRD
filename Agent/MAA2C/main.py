from maa2c import MAA2C

from multiagent.environment import MultiAgentEnv
import multiagent.scenarios as scenarios

import numpy as np

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
	
	test_num = "crossing_fully_coop" 
	env_name = "crossing_fully_coop"
	# experiment_type = "shared" #"prd_above_threshold_ascend" # prd_above_threshold_ascend, greedy, shared
	experiment_type = "prd_above_threshold_ascend"

	episodes_test = [10000*(i+1) for i in range(20)]

	# actor_file_names = ['23-08-2021_PN_ATN_FCN_lr0.0005VN_SAT_FCN_lr0.001_GradNorm0.5_Entropy0.004_trace_decay0.98topK_0select_above_threshold0.0l1_pen0.0critic_entropy_pen0.0_epsiode'+str(e)+'.pt' for e in episodes_test]



	# critic_file_names = []





	# load_path_actors =  ['variance_grad_plots/shared/'+str(test_num)+'/actor_networks/'+a_name for a_name in actor_file_names]
	# load_path_critics = ['variance_grad_plots/shared/'+str(test_num)+'/actor_networks/'+c_name for c_name in critic_file_names]

	# crossing_greedy/ crossing_fully_coop /  paired_by_sharing_goals/ crossing_partially_coop/ color_social_dilemma
	grad_vars = []
	for i in range(len(episodes_test)):
		episode = episodes_test[i]
		extension = "MAA2C_run_"+str(i)
		
		# for crossing_fully_coop
		load_path_actor  =  'variance_grad_plots/shared/'+str(test_num)+'/actor_networks/23-08-2021_PN_ATN_FCN_lr0.0005VN_SAT_FCN_lr0.001_GradNorm0.5_Entropy0.004_trace_decay0.98topK_0select_above_threshold0.0l1_pen0.0critic_entropy_pen0.0_epsiode'+str(episode)+'.pt'
		load_path_critic =  'variance_grad_plots/shared/'+str(test_num)+'/actor_networks/23-08-2021_PN_ATN_FCN_lr0.0005VN_SAT_FCN_lr0.001_GradNorm0.5_Entropy0.004_trace_decay0.98topK_0select_above_threshold0.0l1_pen0.0critic_entropy_pen0.0_epsiode'+str(episode)+'.pt'

		#
		# load_path_actor =02-09-2021_PN_ATN_FCN_lr0.0005VN_SAT_FCN_lr0.001_GradNorm0.5_Entropy0.008_trace_decay0.98topK_0select_above_threshold0.0l1_pen0.0critic_entropy_pen0.0_epsiode1000

		dictionary = {
				"policy_type": "MLP", # MLP/ GAT
				"policy_attention_heads": 0,
				"critic_type": "Transformer",
				"critic_attention_heads": 0,
				"critic_dir": '../../../tests/MAA2C_'+env_name+"_"+experiment_type+'/models/critic_networks/run'+str(i),
				"actor_dir": '../../../tests/MAA2C_'+env_name+"_"+experiment_type+'/models/actor_networks/run'+str(i),
				"gif_dir": '../../../tests/MAA2C_'+env_name+"_"+experiment_type+'/gifs/',
				"policy_eval_dir":'../../../tests/MAA2C_'+env_name+"_"+experiment_type+'/policy_eval_dir/run'+str(i),
				"env": env_name, 
				"test_num":test_num,
				"extension":extension,
				"iteration": 1,
				"device": "gpu",
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
				"threshold_max": 0.03,
				"steps_to_take": 1000, 
				"l1_pen_min": 0.0,
				"l1_pen_steps_to_take": 0,
				"top_k": 0,
				"gif": False,
				"gif_checkpoint":1,
				"load_models": True,
				"model_path_value": " ",
				"model_path_policy": " ",
				"eval_policy": True,
				"save_model": True,
				"save_model_checkpoint": 1000,
				"save_comet_ml_plot": True,
				"learn":True,
				"max_episodes": 100,
				"max_time_steps": 100,
				"experiment_type": experiment_type,
				"gae": True,
				"norm_adv": False,
				"norm_rew": False,
				"load_path_actor":load_path_actor,
				"load_path_critic":load_path_critic
			}

		env = make_env(scenario_name=dictionary["env"],benchmark=False)
		ma_controller = MAA2C(env,dictionary)
		grad_var = ma_controller.run_gradvar_exp(episode)
		print('grad_var: ', grad_var)
		grad_vars.append(grad_var)
		np.save('grad_vars_'+experiment_type+'_'+env_name,np.array(grad_vars))