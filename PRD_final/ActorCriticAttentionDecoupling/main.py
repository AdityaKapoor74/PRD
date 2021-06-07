from maa2c import MAA2C

from multiagent.environment import MultiAgentEnv
# from multiagent.scenarios.simple_spread import Scenario
import multiagent.scenarios as scenarios
import torch 
import numpy as np
import argparse

def make_env(scenario_name, benchmark=False):
	# load scenario from script
	scenario = scenarios.load(scenario_name + ".py").Scenario()
	# scenario = Scenario()
	# create world
	world = scenario.make_world()
	# create multiagent environment
	if benchmark:
		env = MultiAgentEnv(world, scenario.reset_world, scenario.reward_paired_agents, scenario.observation, scenario.benchmark_data, scenario.isFinished)
	else:
		env = MultiAgentEnv(world, scenario.reset_world, scenario.reward_paired_agents, scenario.observation, None, scenario.isFinished)
	return env


def run_file(dictionary):
	env = make_env(scenario_name=dictionary["env"],benchmark=False)
	ma_controller = MAA2C(env,dictionary)
	ma_controller.run()


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("--environment", default="paired_by_sharing_goals", type=str) # paired_by_sharing_goals, collision_avoidance, multi_circular
	parser.add_argument("--experiment_type", default="with_prd_soft_adv", type=str) # greedy_policy, without_prd, with_prd_topK (k=1,num_agents), with_prd_soft_adv
	parser.add_argument("--save", default=False , type=bool)
	parser.add_argument("--max_episodes", default=100000, type=int)
	parser.add_argument("--max_time_steps", default=100, type=int)
	parser.add_argument("--value_lr", default=1e-2, type=float)
	parser.add_argument("--tau", default=1e-3, type=float)
	parser.add_argument("--policy_lr", default=1e-3, type=float)
	parser.add_argument("--entropy_pen", default=8e-3, type=float)
	parser.add_argument("--trace_decay", default=0.98, type=float)
	parser.add_argument("--gamma", default=0.99, type=float)
	parser.add_argument("--select_above_threshold", default=0.1, type=float)
	parser.add_argument("--softmax_cut_threshold", default=0.1, type=float)
	parser.add_argument("--top_k", default=0 , type=int)
	parser.add_argument("--gif", default=False , type=bool)
	parser.add_argument("--store_model", default= "../../../models/", type=str)
	parser.add_argument("--save_runs", default= "../../../runs/", type=str)
	parser.add_argument("--save_gifs", default= "../../../gifs/", type=str)
	parser.add_argument("--td_lambda", default= 0.8, type=float)
	parser.add_argument("--critic_loss_type", default= "td_lambda", type=str) # monte_carlo, td_lambda, td_1
	parser.add_argument("--critic_update_type", default= "soft", type=str) # soft, hard
	parser.add_argument("--critic_update_interval", default= 100, type=int)
	parser.add_argument("--gae", default= True, type=bool)
	parser.add_argument("--norm_adv", default= False, type=bool)
	parser.add_argument("--norm_rew", default= False, type=bool)
	parser.add_argument("--load_models", default= False, type=bool)
	parser.add_argument("--critic_saved_path", default= "", type=str)
	parser.add_argument("--actor_saved_path", default= "", type=str)
	parser.add_argument("--l1_pen", default= 0.01, type=float)
	parser.add_argument("--anneal_entropy_pen", default= False, type=bool)
	parser.add_argument("--entropy_pen_end", default= 0.0, type=float)
	parser.add_argument("--entropy_pen_decay", default= 0.0, type=float)
	

	arguments = parser.parse_args()

	if arguments.environment in ["paired_by_sharing_goals", "collision_avoidance"] :
		extender = "/16_Agents"
	elif arguments.environment == "multi_circular":
		extender = "/4_Agents_2_Circles_2_Agents_Per_Circle"

	dictionary = {
		"critic_dir": arguments.store_model+'/'+arguments.environment+'/'+arguments.experiment_type+extender+'/critic_networks/',
		"actor_dir": arguments.store_model+'/'+arguments.environment+'/'+arguments.experiment_type+extender+'/actor_networks/',
		"tensorboard_dir": arguments.save_runs+'/'+arguments.environment+'/'+arguments.experiment_type+extender+'/',
		"gif_dir": arguments.save_gifs+'/'+arguments.environment+'/'+arguments.experiment_type+extender+'/',
		"env": arguments.environment, 
		"value_lr": arguments.value_lr,
		"tau": arguments.tau,
		"policy_lr": arguments.policy_lr,
		"entropy_pen": arguments.entropy_pen, 
		"gamma": arguments.gamma,
		"trace_decay": arguments.trace_decay,
		"select_above_threshold": arguments.select_above_threshold,
		"softmax_cut_threshold": arguments.softmax_cut_threshold,
		"top_k": arguments.top_k,
		"gif": arguments.gif,
		"save": arguments.save,
		"max_episodes": arguments.max_episodes,
		"max_time_steps": arguments.max_time_steps,
		"experiment_type": arguments.experiment_type,
		"td_lambda": arguments.td_lambda,
		"critic_loss_type": arguments.critic_loss_type,
		"critic_update_type": arguments.critic_update_type,
		"gae": arguments.gae,
		"norm_adv": arguments.norm_adv,
		"norm_rew": arguments.norm_rew,
		"load_models": arguments.load_models,
		"critic_saved_path": arguments.critic_saved_path,
		"actor_saved_path": arguments.actor_saved_path,
		"l1_pen": arguments.l1_pen, 
		"anneal_entropy_pen": arguments.anneal_entropy_pen,
		"entropy_pen_end": arguments.entropy_pen_end,
		"entropy_pen_decay": arguments.entropy_pen_decay,
		"critic_update_interval": arguments.critic_update_interval
	}

	
	env = make_env(scenario_name=dictionary["env"],benchmark=False)
	ma_controller = MAA2C(env,dictionary)
	ma_controller.run()