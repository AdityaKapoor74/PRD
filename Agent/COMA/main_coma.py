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
	return env


def run_file(dictionary):
	env = make_env(scenario_name=dictionary["env"],benchmark=False)
	ma_controller = MAA2C(env,dictionary)
	ma_controller.run()


if __name__ == '__main__':

	for i in range(1,6):
		extension = "COMA_GNN_run_"+str(i)
		test_num = "COMA_GNN" #TransformersTest
		env_name = "paired_by_sharing_goals"

		dictionary = {
				"policy_type": "MLP", # MLP/ GAT
				"policy_attention_heads": 0,
				"critic_type": "Transformer", # Transformer/ DualTransformer
				"critic_attention_heads": 0,
				"critic_dir": " ",
				"actor_dir": " ",
				"gif_dir": " ",
				"policy_eval_dir":'',
				"env": env_name, 
				"test_num":test_num,
				"extension":extension,
				"value_lr": 5e-4, 
				"policy_lr": 5e-4,
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
				"max_episodes": 20000,
				"max_time_steps": 100,
				"norm_adv": False,
				"norm_rew": False,
			}
		env = make_env(scenario_name=dictionary["env"],benchmark=False)
		ma_controller = MACOMA(env,dictionary)
		ma_controller.run()
