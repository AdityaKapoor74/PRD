import os
import sys

from main_collision_avoidance import run_file



if __name__ == '__main__':
	env_list = ["collision_avoidance"]

	experiment_type = ["greedy_policy", "without_prd", "with_prd_top1", "with_prd_top3", "with_prd_soft_adv", "without_prd_scaled", "with_prd_top1_scaled", "with_prd_top3_scaled", "with_prd_soft_adv_scaled"]

	top_k_list = [0, 1, 3, 0, 0, 1, 3, 0]
	
	for i in range(len(env_list)):
		for j in range(len(experiment_type)):
			dictionary = {
				"critic_dir": '../../../run_other_collision_rew/models/Scalar_dot_product/'+env_list[i]+'/6_Agents/SingleAttentionMechanism/'+experiment_type[j]+'/critic_networks/',
				"actor_dir": '../../../run_other_collision_rew/models/Scalar_dot_product/'+env_list[i]+'/6_Agents/SingleAttentionMechanism/'+experiment_type[j]+'/actor_networks/',
				"tensorboard_dir":'../../../run_other_collision_rew/runs/Scalar_dot_product/'+env_list[i]+'/6_Agents/SingleAttentionMechanism/'+experiment_type[j]+'/',
				"gif_dir": '../../../gifs/Scalar_dot_product/'+env_list[i]+'/6_Agents/SingleAttentionMechanism/'+experiment_type[j]+'/',
				"env": env_list[i], 
				"value_lr": 1e-2, #1e-2 for single head
				"policy_lr": 1e-3, # 2e-4 for single head
				"entropy_pen": 0.008, 
				"gamma": 0.99,
				"trace_decay": 0.98,
				"select_above_threshold": 0.1,
				"softmax_cut_threshold": 0.1,
				"experiment_type": experiment_type[j],
				"top_k": top_k_list[j],
				"gif": False,
				"save": True,
				"max_episodes": 80000,
				"max_time_steps": 100,
			}

			run_file(dictionary)


# RUN GREEDY 5 times
# if __name__ == '__main__':
# 	env_list = ["collision_avoidance"]

# 	experiment_type = ["greedy_policy", "greedy_policy", "greedy_policy", "greedy_policy", "greedy_policy"]

# 	top_k_list = [0, 0, 0, 0, 0]
	
# 	for i in range(len(env_list)):
# 		for index,experiment in enumerate(experiment_type):
# 			dictionary = {
# 				"critic_dir": '../../../run'+str(index)+'/models/Scalar_dot_product/'+env_list[i]+'/6_Agents/SingleAttentionMechanism/'+experiment+'/critic_networks/',
# 				"actor_dir": '../../../run'+str(index)+'/models/Scalar_dot_product/'+env_list[i]+'/6_Agents/SingleAttentionMechanism/'+experiment+'/actor_networks/',
# 				"tensorboard_dir":'../../../run'+str(index)+'/runs/Scalar_dot_product/'+env_list[i]+'/6_Agents/SingleAttentionMechanism/'+experiment+'/',
# 				"gif_dir": '../../../gifs/Scalar_dot_product/'+env_list[i]+'/6_Agents/SingleAttentionMechanism/'+experiment+'/',
# 				"env": env_list[i], 
# 				"value_lr": 1e-2, #1e-2 for single head
# 				"policy_lr": 1e-3, # 2e-4 for single head
# 				"entropy_pen": 0.008, 
# 				"gamma": 0.99,
# 				"trace_decay": 0.98,
# 				"select_above_threshold": 0.1,
# 				"softmax_cut_threshold": 0.1,
# 				"experiment_type": experiment,
# 				"top_k": top_k_list[index],
# 				"gif": False,
# 				"save": True,
# 				"max_episodes": 80000,
# 				"max_time_steps": 100,
# 			}

# 			run_file(dictionary)