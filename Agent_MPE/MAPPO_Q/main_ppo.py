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
	ma_controller = MAA2C(env,dictionary)
	ma_controller.run()


if __name__ == '__main__':

	for i in range(1,2):
		extension = "MAPPO_Q_run_"+str(i)
		test_num = "PRD_2_MPE"
		env_name = "crossing_team_greedy" # paired_by_sharing_goals, color_social_dilemma, crossing_team_greedy, crossing_greedy, crossing_partially_coop, crossing_fully_coop
		experiment_type = "prd_above_threshold_ascend"

		dictionary = {
				"iteration": i,
				"grad_clip_critic": 10.0,
				"grad_clip_actor": 10.0,
				"device": "cpu",
				"critic_dir": '../../../tests/'+test_num+'/models/'+env_name+'_'+experiment_type+'_'+extension+'/critic_networks/',
				"actor_dir": '../../../tests/'+test_num+'/models/'+env_name+'_'+experiment_type+'_'+extension+'/actor_networks/',
				"gif_dir": '../../../tests/'+test_num+'/gifs/'+env_name+'_'+experiment_type+'_'+extension+'/',
				"policy_eval_dir":'../../../tests/'+test_num+'/policy_eval/'+env_name+'_'+experiment_type+'_'+extension+'/',
				"policy_clip": 0.05,
				"value_clip": 0.05,
				"n_epochs": 5,
				"update_ppo_agent": 1, # update ppo agent after every update_ppo_agent episodes
				"env": env_name, 
				"test_num":test_num,
				"extension":extension,
				"value_lr": 1e-3, #1e-3
				"policy_lr": 7e-4, #prd 1e-4
				"entropy_pen": 8e-3, #8e-3
				"gamma": 0.99, 
				"gae_lambda": 0.95,
				"lambda": 0.95, # 1 --> Monte Carlo; 0 --> TD(1)
				"select_above_threshold": 0.0,
				"threshold_min": 0.0, 
				"threshold_max": 0.2,
				"steps_to_take": 1000,
				"top_k": 0,
				"gif": False,
				"gif_checkpoint":1,
				"load_models": False,
				"model_path_value": "../../../tests/PRD_2_MPE/models/crossing_team_greedy_shared_MAPPO_Q_run_1/critic_networks/critic_epsiode11000.pt",
				"model_path_policy": "../../../tests/PRD_2_MPE/models/crossing_team_greedy_shared_MAPPO_Q_run_1/actor_networks/actor_epsiode11000.pt",
				"eval_policy": True,
				"save_model": True,
				"save_model_checkpoint": 1000,
				"save_comet_ml_plot": True,
				"learn":True,
				"max_episodes": 200000,
				"max_time_steps": 50,
				"experiment_type": experiment_type,
				"norm_adv": False,
				"norm_returns": False,
				"value_normalization": False,
				"parallel_training": False,
			}
		env = make_env(scenario_name=dictionary["env"],benchmark=False)
		ma_controller = MAPPO(env,dictionary)
		ma_controller.run()