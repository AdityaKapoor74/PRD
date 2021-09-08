import numpy as np
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

	# color_social_dilemma
	# for i in range(1,3):
	# 	extension = "run"+str(i)
	# 	test_num = "color_social_dilemma_32_Agents_200K_policy_eval"
	# 	env_name = "color_social_dilemma" 
	# 	experiment_type = "shared" # prd_above_threshold_decay, greedy, shared, prd_above_threshold_ascend, prd_above_threshold_l1_pen_decay

	# 	dictionary = {
	# 			"critic_dir": '../../../tests/'+test_num+'/models/'+env_name+'_'+experiment_type+'_'+extension+'/critic_networks/',
	# 			"actor_dir": '../../../tests/'+test_num+'/models/'+env_name+'_'+experiment_type+'_'+extension+'/actor_networks/',
	# 			"run_dir":'../../../tests/'+test_num+'/runs/'+env_name+'_'+experiment_type+'_'+extension+'/',
	# 			"policy_eval_dir":'../../../tests/'+test_num+'/policy_eval/'+env_name+'_'+experiment_type+'_'+extension+'/',
	# 			"gif_dir": '../../../tests/'+test_num+'/gifs/'+env_name+'_'+experiment_type+'_'+extension+'/',
	# 			"env": env_name, 
	# 			"test_num":test_num,
	# 			"value_lr": 1e-3, 
	# 			"policy_lr": 2e-4, # 5e-4(shared) 8e-5(prd_above_threshold_ascend)
	# 			"entropy_pen": 8e-3, 
	# 			"entropy_pen_min": 8e-3,
	# 			"l1_pen": 0.0,
	# 			"critic_entropy_pen": 0.0,
	# 			"critic_loss_type": "TD_lambda", #MC
	# 			"gamma": 0.99, 
	# 			"trace_decay": 0.98,
	# 			"lambda": 0.8, #0.8
	# 			"select_above_threshold": 0.0,
	# 			"threshold_min": 0.0, #0.0001
	# 			"threshold_max": 0.01,
	# 			"steps_to_take": 15000,
	# 			"l1_pen_min": 0.0,
	# 			"l1_pen_steps_to_take": 0,
	# 			"top_k": 0,
	# 			"gif": False,
	# 			"save_model": True,
	# 			"eval_policy": True,
	# 			"save_model_checkpoint": 100,
	# 			"save_tensorboard_plot": False,
	# 			"save_comet_ml_plot": True,
	# 			"learn":True,
	# 			"max_episodes": 200000,
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


	# crossing_greedy/ crossing_fully_coop /  paired_by_sharing_goals/ crossing_partially_coop
	# for i in range(1,4):

	actor_file_names = [
# '12-08-2021_PN_ATN_FCN_lr0.0005VN_SAT_FCN_lr0.001_GradNorm0.5_Entropy0.008_trace_decay0.98topK_0select_above_threshold0.0l1_pen0.0critic_entropy_pen0.0_epsiode1000.pt',
# '12-08-2021_PN_ATN_FCN_lr0.0005VN_SAT_FCN_lr0.001_GradNorm0.5_Entropy0.008_trace_decay0.98topK_0select_above_threshold0.0l1_pen0.0critic_entropy_pen0.0_epsiode2000.pt',
# '12-08-2021_PN_ATN_FCN_lr0.0005VN_SAT_FCN_lr0.001_GradNorm0.5_Entropy0.008_trace_decay0.98topK_0select_above_threshold0.0l1_pen0.0critic_entropy_pen0.0_epsiode3000.pt',
# '12-08-2021_PN_ATN_FCN_lr0.0005VN_SAT_FCN_lr0.001_GradNorm0.5_Entropy0.008_trace_decay0.98topK_0select_above_threshold0.0l1_pen0.0critic_entropy_pen0.0_epsiode4000.pt',
# '12-08-2021_PN_ATN_FCN_lr0.0005VN_SAT_FCN_lr0.001_GradNorm0.5_Entropy0.008_trace_decay0.98topK_0select_above_threshold0.0l1_pen0.0critic_entropy_pen0.0_epsiode5000.pt',
# '12-08-2021_PN_ATN_FCN_lr0.0005VN_SAT_FCN_lr0.001_GradNorm0.5_Entropy0.008_trace_decay0.98topK_0select_above_threshold0.0l1_pen0.0critic_entropy_pen0.0_epsiode6000.pt',
# '12-08-2021_PN_ATN_FCN_lr0.0005VN_SAT_FCN_lr0.001_GradNorm0.5_Entropy0.008_trace_decay0.98topK_0select_above_threshold0.0l1_pen0.0critic_entropy_pen0.0_epsiode7000.pt',
'12-08-2021_PN_ATN_FCN_lr0.0005VN_SAT_FCN_lr0.001_GradNorm0.5_Entropy0.008_trace_decay0.98topK_0select_above_threshold0.0l1_pen0.0critic_entropy_pen0.0_epsiode8000.pt',
'12-08-2021_PN_ATN_FCN_lr0.0005VN_SAT_FCN_lr0.001_GradNorm0.5_Entropy0.008_trace_decay0.98topK_0select_above_threshold0.0l1_pen0.0critic_entropy_pen0.0_epsiode9000.pt',
# '12-08-2021_PN_ATN_FCN_lr0.0005VN_SAT_FCN_lr0.001_GradNorm0.5_Entropy0.008_trace_decay0.98topK_0select_above_threshold0.0l1_pen0.0critic_entropy_pen0.0_epsiode10000.pt',
# '12-08-2021_PN_ATN_FCN_lr0.0005VN_SAT_FCN_lr0.001_GradNorm0.5_Entropy0.008_trace_decay0.98topK_0select_above_threshold0.0l1_pen0.0critic_entropy_pen0.0_epsiode11000.pt',
# '12-08-2021_PN_ATN_FCN_lr0.0005VN_SAT_FCN_lr0.001_GradNorm0.5_Entropy0.008_trace_decay0.98topK_0select_above_threshold0.0l1_pen0.0critic_entropy_pen0.0_epsiode12000.pt',
# '12-08-2021_PN_ATN_FCN_lr0.0005VN_SAT_FCN_lr0.001_GradNorm0.5_Entropy0.008_trace_decay0.98topK_0select_above_threshold0.0l1_pen0.0critic_entropy_pen0.0_epsiode13000.pt',
# '12-08-2021_PN_ATN_FCN_lr0.0005VN_SAT_FCN_lr0.001_GradNorm0.5_Entropy0.008_trace_decay0.98topK_0select_above_threshold0.0l1_pen0.0critic_entropy_pen0.0_epsiode14000.pt',
# '12-08-2021_PN_ATN_FCN_lr0.0005VN_SAT_FCN_lr0.001_GradNorm0.5_Entropy0.008_trace_decay0.98topK_0select_above_threshold0.0l1_pen0.0critic_entropy_pen0.0_epsiode15000.pt',
# '12-08-2021_PN_ATN_FCN_lr0.0005VN_SAT_FCN_lr0.001_GradNorm0.5_Entropy0.008_trace_decay0.98topK_0select_above_threshold0.0l1_pen0.0critic_entropy_pen0.0_epsiode16000.pt',
# '12-08-2021_PN_ATN_FCN_lr0.0005VN_SAT_FCN_lr0.001_GradNorm0.5_Entropy0.008_trace_decay0.98topK_0select_above_threshold0.0l1_pen0.0critic_entropy_pen0.0_epsiode17000.pt',
# '12-08-2021_PN_ATN_FCN_lr0.0005VN_SAT_FCN_lr0.001_GradNorm0.5_Entropy0.008_trace_decay0.98topK_0select_above_threshold0.0l1_pen0.0critic_entropy_pen0.0_epsiode18000.pt',
# '12-08-2021_PN_ATN_FCN_lr0.0005VN_SAT_FCN_lr0.001_GradNorm0.5_Entropy0.008_trace_decay0.98topK_0select_above_threshold0.0l1_pen0.0critic_entropy_pen0.0_epsiode19000.pt',
# '12-08-2021_PN_ATN_FCN_lr0.0005VN_SAT_FCN_lr0.001_GradNorm0.5_Entropy0.008_trace_decay0.98topK_0select_above_threshold0.0l1_pen0.0critic_entropy_pen0.0_epsiode20000.pt',
]

	critic_file_names = [
# '12-08-2021VN_ATN_FCN_lr0.001_PN_ATN_FCN_lr0.0005_GradNorm0.5_Entropy0.008_trace_decay0.98topK_0select_above_threshold0.0l1_pen0.0critic_entropy_pen0.0_epsiode1000.pt',
# '12-08-2021VN_ATN_FCN_lr0.001_PN_ATN_FCN_lr0.0005_GradNorm0.5_Entropy0.008_trace_decay0.98topK_0select_above_threshold0.0l1_pen0.0critic_entropy_pen0.0_epsiode2000.pt',
# '12-08-2021VN_ATN_FCN_lr0.001_PN_ATN_FCN_lr0.0005_GradNorm0.5_Entropy0.008_trace_decay0.98topK_0select_above_threshold0.0l1_pen0.0critic_entropy_pen0.0_epsiode3000.pt',
# '12-08-2021VN_ATN_FCN_lr0.001_PN_ATN_FCN_lr0.0005_GradNorm0.5_Entropy0.008_trace_decay0.98topK_0select_above_threshold0.0l1_pen0.0critic_entropy_pen0.0_epsiode4000.pt',
# '12-08-2021VN_ATN_FCN_lr0.001_PN_ATN_FCN_lr0.0005_GradNorm0.5_Entropy0.008_trace_decay0.98topK_0select_above_threshold0.0l1_pen0.0critic_entropy_pen0.0_epsiode5000.pt',
# '12-08-2021VN_ATN_FCN_lr0.001_PN_ATN_FCN_lr0.0005_GradNorm0.5_Entropy0.008_trace_decay0.98topK_0select_above_threshold0.0l1_pen0.0critic_entropy_pen0.0_epsiode6000.pt',
# '12-08-2021VN_ATN_FCN_lr0.001_PN_ATN_FCN_lr0.0005_GradNorm0.5_Entropy0.008_trace_decay0.98topK_0select_above_threshold0.0l1_pen0.0critic_entropy_pen0.0_epsiode7000.pt',
'12-08-2021VN_ATN_FCN_lr0.001_PN_ATN_FCN_lr0.0005_GradNorm0.5_Entropy0.008_trace_decay0.98topK_0select_above_threshold0.0l1_pen0.0critic_entropy_pen0.0_epsiode8000.pt',
'12-08-2021VN_ATN_FCN_lr0.001_PN_ATN_FCN_lr0.0005_GradNorm0.5_Entropy0.008_trace_decay0.98topK_0select_above_threshold0.0l1_pen0.0critic_entropy_pen0.0_epsiode9000.pt',
# '12-08-2021VN_ATN_FCN_lr0.001_PN_ATN_FCN_lr0.0005_GradNorm0.5_Entropy0.008_trace_decay0.98topK_0select_above_threshold0.0l1_pen0.0critic_entropy_pen0.0_epsiode10000.pt',
# '12-08-2021VN_ATN_FCN_lr0.001_PN_ATN_FCN_lr0.0005_GradNorm0.5_Entropy0.008_trace_decay0.98topK_0select_above_threshold0.0l1_pen0.0critic_entropy_pen0.0_epsiode11000.pt',
# '12-08-2021VN_ATN_FCN_lr0.001_PN_ATN_FCN_lr0.0005_GradNorm0.5_Entropy0.008_trace_decay0.98topK_0select_above_threshold0.0l1_pen0.0critic_entropy_pen0.0_epsiode12000.pt',
# '12-08-2021VN_ATN_FCN_lr0.001_PN_ATN_FCN_lr0.0005_GradNorm0.5_Entropy0.008_trace_decay0.98topK_0select_above_threshold0.0l1_pen0.0critic_entropy_pen0.0_epsiode13000.pt',
# '12-08-2021VN_ATN_FCN_lr0.001_PN_ATN_FCN_lr0.0005_GradNorm0.5_Entropy0.008_trace_decay0.98topK_0select_above_threshold0.0l1_pen0.0critic_entropy_pen0.0_epsiode14000.pt',
# '12-08-2021VN_ATN_FCN_lr0.001_PN_ATN_FCN_lr0.0005_GradNorm0.5_Entropy0.008_trace_decay0.98topK_0select_above_threshold0.0l1_pen0.0critic_entropy_pen0.0_epsiode15000.pt',
# '12-08-2021VN_ATN_FCN_lr0.001_PN_ATN_FCN_lr0.0005_GradNorm0.5_Entropy0.008_trace_decay0.98topK_0select_above_threshold0.0l1_pen0.0critic_entropy_pen0.0_epsiode16000.pt',
# '12-08-2021VN_ATN_FCN_lr0.001_PN_ATN_FCN_lr0.0005_GradNorm0.5_Entropy0.008_trace_decay0.98topK_0select_above_threshold0.0l1_pen0.0critic_entropy_pen0.0_epsiode17000.pt',
# '12-08-2021VN_ATN_FCN_lr0.001_PN_ATN_FCN_lr0.0005_GradNorm0.5_Entropy0.008_trace_decay0.98topK_0select_above_threshold0.0l1_pen0.0critic_entropy_pen0.0_epsiode18000.pt',
# '12-08-2021VN_ATN_FCN_lr0.001_PN_ATN_FCN_lr0.0005_GradNorm0.5_Entropy0.008_trace_decay0.98topK_0select_above_threshold0.0l1_pen0.0critic_entropy_pen0.0_epsiode19000.pt',
# '12-08-2021VN_ATN_FCN_lr0.001_PN_ATN_FCN_lr0.0005_GradNorm0.5_Entropy0.008_trace_decay0.98topK_0select_above_threshold0.0l1_pen0.0critic_entropy_pen0.0_epsiode20000.pt'
]

	load_path_actors = ['paired_agent_model/paired_by_sharing_goals_shared_run1/actor_networks/'+a_name for a_name in actor_file_names]
	load_path_critics = ['paired_agent_model/paired_by_sharing_goals_shared_run1/critic_networks/'+c_name for c_name in critic_file_names]
	thresholds = [0.00066667,0.00133333,0.002,0.00266667,.00333333,0.004,0.00466667,0.00533333,0.006,0.00666667,0.00733333,0.008,0.00866667,0.00933333,0.01,0.01,0.01,0.01,0.01,0.01]
	assert len(thresholds) >= len(load_path_actors)


	grad_vars = []
	for i in range(len(load_path_actors)):
		extension = "run"+str(i)
		test_num = "paired_by_sharing_goals_30_agents" #crossing_8_agents_pen_non_colliding_agents_policy_eval
		env_name = "paired_by_sharing_goals"
		experiment_type = "shared" # prd_above_threshold_decay_episodic, greedy, shared
		# experiment_type = "prd_above_threshold"

		dictionary = {
				"critic_dir": '../../../tests/'+test_num+'/models/'+env_name+'_'+experiment_type+'_'+extension+'/critic_networks/',
				"actor_dir": '../../../tests/'+test_num+'/models/'+env_name+'_'+experiment_type+'_'+extension+'/actor_networks/',
				"run_dir":'../../../tests/'+test_num+'/runs/'+env_name+'_'+experiment_type+'_'+extension+'/',
				"gif_dir": '../../../tests/'+test_num+'/gifs/'+env_name+'_'+experiment_type+'_'+extension+'/',
				"policy_eval_dir":'../../../tests/'+test_num+'/policy_eval/'+env_name+'_'+experiment_type+'_'+extension+'/',
				"env": env_name, 
				"test_num":test_num,
				"extension":extension,
				"value_lr": 1e-3, 
				"policy_lr": 5e-4, #prd 1e-4
				"entropy_pen": 8e-3, 
				"entropy_pen_min": 8e-3,
				"l1_pen": 0.0,
				"critic_entropy_pen": 0.0,
				"critic_loss_type": "TD_lambda",
				"gamma": 0.99, 
				"trace_decay": 0.98,
				"lambda": 0.8, #0.8
				"select_above_threshold": thresholds[i],
				"threshold_min": 0.0, 
				"threshold_max": 0.01,
				"steps_to_take": 15000, 
				"l1_pen_min": 0.0,
				"l1_pen_steps_to_take": 0,
				"top_k": 0,
				"gif": False,
				"eval_policy": True,
				"save_model": True,
				"save_model_checkpoint": 1000,
				"save_tensorboard_plot": False,
				"save_comet_ml_plot": True,
				"learn":True,
				"max_episodes": 100,
				"max_time_steps": 100,
				"experiment_type": experiment_type,
				"gif_checkpoint":1,
				"gae": True,
				"norm_adv": False,
				"norm_rew": False,
				"load_path_actor":load_path_actors[i],
				"load_path_critic":load_path_critics[i]

			}
		env = make_env(scenario_name=dictionary["env"],benchmark=False)
		ma_controller = MAA2C(env,dictionary)
		grad_var = ma_controller.run_gradvar_exp()
		print('grad_var: ', grad_var)
		grad_vars.append(grad_var)
		np.save('grad_vars_'+str(experiment_type)+'_8_and_9k',np.array(grad_vars))


# export CUDA_VISIBLE_DEVICES=1
# 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

# 1) ssh biorobotics@highbay.pc.cs.cmu.edu
# Ben
# password: biorobotics
# Ben
# 2) ssh biorobotics@10.0.0.8
# Ben
# Ben Freed
# password: biorobotics
