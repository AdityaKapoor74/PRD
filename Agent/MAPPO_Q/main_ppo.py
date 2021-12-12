from mappo import MAPPO

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
	ma_controller = MAPPO(env,dictionary)
	ma_controller.run()


if __name__ == '__main__':

	for i in range(1,2):
		extension = "MAPPO_run_"+str(i)
		test_num = "PRD_2_exps"
		env_name = "crossing_team_greedy" # paired_by_sharing_goals, color_social_dilemma, crossing_team_greedy, crossing_greedy, crossing_partially_coop, crossing_fully_coop
		experiment_type = "prd_above_threshold_ascend"

		dictionary = {
				"iteration": i,
				"grad_clip_critic": 10,
				"grad_clip_actor": 10,
				"device": "gpu",
				"critic_dir": '../../../tests/'+test_num+'/models/'+env_name+'_'+experiment_type+'_'+extension+'/critic_networks/',
				"actor_dir": '../../../tests/'+test_num+'/models/'+env_name+'_'+experiment_type+'_'+extension+'/actor_networks/',
				"gif_dir": '../../../tests/'+test_num+'/gifs/'+env_name+'_'+experiment_type+'_'+extension+'/',
				"policy_eval_dir":'../../../tests/'+test_num+'/policy_eval/'+env_name+'_'+experiment_type+'_'+extension+'/',
				"policy_clip": 0.2,
				"n_epochs": 5,
				"update_ppo_agent": 1, # update ppo agent after every update_ppo_agent episodes
				"env": env_name, 
				"test_num":test_num,
				"extension":extension,
				"pen_threshold": 0.01,
				"value_lr": 1e-3, #1e-3
				"policy_lr": 1e-4, #prd 1e-4
				"entropy_pen": 0.0, #8e-3
				"gamma": 0.99, 
				"trace_decay": 0.95,
				"lambda": 0.8, #0.8
				"select_above_threshold": 0.0,
				"threshold_min": 0.0, 
				"threshold_max": 0.05,
				"steps_to_take": 1000,
				"top_k": 0,
				"gif": False,
				"gif_checkpoint":1,
				"load_models": False,
				"model_path_value": "../../../tests/PRD_2_exps/models/crossing_team_greedy_prd_above_threshold_ascend_MAPPO_run_1/critic_networks/04-12-2021VN_ATN_FCN_lr0.001_PN_ATN_FCN_lr0.0001_GradNorm0.5_Entropy0.008_trace_decay0.98topK_0select_above_threshold0.0l1_pen0.0critic_entropy_pen0.0_epsiode10000.pt",
				"model_path_policy": "../../../tests/PRD_2_exps/models/crossing_team_greedy_prd_above_threshold_ascend_MAPPO_run_1/actor_networks/04-12-2021_PN_ATN_FCN_lr0.0001VN_SAT_FCN_lr0.001_GradNorm0.5_Entropy0.008_trace_decay0.98topK_0select_above_threshold0.0l1_pen0.0critic_entropy_pen0.0_epsiode10000.pt",
				"eval_policy": False,
				"save_model": False,
				"save_model_checkpoint": 1000,
				"save_comet_ml_plot": True,
				"learn":True,
				"max_episodes": 100000,
				"max_time_steps": 100,
				"experiment_type": experiment_type,
				"norm_adv": False,
				"norm_returns": False,
			}
		env = make_env(scenario_name=dictionary["env"],benchmark=False)
		ma_controller = MAPPO(env,dictionary)
		ma_controller.run()