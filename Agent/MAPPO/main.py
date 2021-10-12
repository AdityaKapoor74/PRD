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
		extension = "MAPPO"+str(i)
		test_num = "MAPPO_exps"
		env_name = "crossing_team_greedy" # paired_by_sharing_goals, color_social_dilemma, crossing
		experiment_type = "prd_soft_adv"

		dictionary = {
				"policy_type": "MLP", # MLP/ GCN/ GAT
				"policy_attention_heads": 0,
				"critic_type": "Transformer", # TransformersONLY/ GATONLY/ GATv2ONLY/ NormalizedATONLY/ else One Critic for training
				"critic_attention_heads": 0,
				"critic_dir": '../../../tests/'+test_num+'/models/'+env_name+'_'+experiment_type+'_'+extension+'/critic_networks/',
				"actor_dir": '../../../tests/'+test_num+'/models/'+env_name+'_'+experiment_type+'_'+extension+'/actor_networks/',
				"run_dir":'../../../tests/'+test_num+'/runs/'+env_name+'_'+experiment_type+'_'+extension+'/',
				"gif_dir": '../../../tests/ablation_study/'+test_num+'/gifs/'+env_name+'_'+experiment_type+'_'+extension+'/',
				"policy_eval_dir":'../../../tests/'+test_num+'/policy_eval/'+env_name+'_'+experiment_type+'_'+extension+'/',
				"policy_clip": 0.1,
				"n_epochs": 10,
				"update_ppo_agent": 1, # update ppo agent after every 4 episodes
				"env": env_name, 
				"test_num":test_num,
				"extension":extension,
				"value_lr": 1e-3, 
				"policy_lr": 1e-4, #prd 1e-4
				"entropy_pen": 8e-3, 
				"entropy_pen_min": 8e-3,
				"l1_pen": 0.0,
				"critic_entropy_pen": 0.0,
				"critic_loss_type": "TD_lambda",
				"gamma": 0.99, 
				"trace_decay": 0.98,
				"lambda": 0.8, #0.8
				"select_above_threshold": 0.0,
				"threshold_min": 0.0, 
				"threshold_max": 0.0,
				"steps_to_take": 1000, 
				"l1_pen_min": 0.0,
				"l1_pen_steps_to_take": 0,
				"top_k": 0,
				"gif": False,
				"gif_checkpoint":1,
				"load_models": False,
				"model_path_value": "../../../tests/ablation_study/Transformers/tests/MultiHeadTransformerCritic2/models/crossing_team_greedy_shared_run1/critic_networks/24-09-2021VN_ATN_FCN_lr0.001_PN_ATN_FCN_lr0.0001_GradNorm0.5_Entropy0.008_trace_decay0.98topK_0select_above_threshold0.0l1_pen0.0critic_entropy_pen0.0_epsiode100000.pt",
				"model_path_policy": "../../../tests/ablation_study/Transformers/tests/MultiHeadTransformerCritic2/models/crossing_team_greedy_shared_run1/actor_networks/24-09-2021_PN_ATN_FCN_lr0.0001VN_SAT_FCN_lr0.001_GradNorm0.5_Entropy0.008_trace_decay0.98topK_0select_above_threshold0.0l1_pen0.0critic_entropy_pen0.0_epsiode100000.pt",
				"eval_policy": False,
				"save_model": True,
				"save_model_checkpoint": 1000,
				"save_comet_ml_plot": True,
				"learn":True,
				"max_episodes": 50000,
				"max_time_steps": 100,
				"experiment_type": experiment_type,
				"gae": True,
				"norm_adv": False,
				"norm_rew": False,
			}
		env = make_env(scenario_name=dictionary["env"],benchmark=False)
		ma_controller = MAPPO(env,dictionary)
		ma_controller.run()

# COLOR SOCIAL DILEMMA PT2
'''
Value Lr 1e-3, 1e-3, 1e-3
Policy Lr 5e-4, 1e-4, 1e-4
Entropy 1e-3, 1e-3, 8e-4
Threshold 5e-3, 5e-3, 5e-3

'''