from macoma import MACOMA

from multiagent.environment import MultiAgentEnv
import multiagent.scenarios as scenarios

def make_env(scenario_name, benchmark=False):
	scenario = scenarios.load(scenario_name + ".py").Scenario()
	world = scenario.make_world()
	if benchmark:
		env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation, scenario.benchmark_data, scenario.isFinished)
	else:
		env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation, None, scenario.isFinished)
	return env, scenario.observation_shape, scenario.transformer_observation_shape


def run_file(dictionary):
	env = make_env(scenario_name=dictionary["env"],benchmark=False)
	ma_controller = MAA2C(env,dictionary)
	ma_controller.run()


if __name__ == '__main__':

	for i in range(1,6):
		extension = "COMA_"+str(i)
		test_num = "TEAM COLLISION AVOIDANCE"
		env_name = "crossing_team_greedy"

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
				"policy_lr": 7e-4,
				"grad_clip_critic": 10.0,
				"grad_clip_actor": 10.0,
				"critic_entropy_pen": 0.0,
				"epsilon_start": 0.5,
				"epsilon_end": 0.02,
				"epsilon_episode_steps": 750,
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
				"max_episodes": 50000,
				"max_time_steps": 100,
				"norm_adv": False,
				"norm_rew": False,
			}
		env, global_observation_shape, local_observation_shape = make_env(scenario_name=dictionary["env"],benchmark=False)
		dictionary["global_observation_shape"] = global_observation_shape
		dictionary["local_observation_shape"] = local_observation_shape
		ma_controller = MACOMA(env,dictionary)
		ma_controller.run()
