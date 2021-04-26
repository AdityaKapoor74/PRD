from maa2c import MAA2C
import imageio
import numpy as np
import os
import torch
from multiagent.environment import MultiAgentEnv
import multiagent.scenarios as scenarios
# os.chdir("/home/aditya/Desktop/Partial_Reward_Decoupling/PRD/SimpleSpread/models_gif")

def make_env(scenario_name, benchmark=False):
	# load scenario from script
	scenario = scenarios.load(scenario_name + ".py").Scenario()
	# scenario = Scenario()
	# create world
	world = scenario.make_world()
	# create multiagent environment
	if benchmark:
		env = MultiAgentEnv(world, scenario.reset_world, scenario.reward_agent_centric_paired, scenario.observation, scenario.benchmark_data, scenario.isFinished)
	else:
		env = MultiAgentEnv(world, scenario.reset_world, scenario.reward_agent_centric_paired, scenario.observation, None, scenario.isFinished)
	return env



if __name__ == '__main__':
	env = make_env(scenario_name="custom_env")
	ma_controller = MAA2C(env,True)




	# Number of images to capture
	n_images = 1000

	images = []

	# init a new episode
	states = env.reset()
	states_critic,states_actor = ma_controller.split_states(states)
	# init the img var with the starting state of the env
	img = env.render(mode='rgb_array')[0]

	for i in range(n_images):
		# At each step, append an image to list
		images.append(img)

		policies = ma_controller.agents.policy_network.forward(torch.FloatTensor(states_actor).to(ma_controller.device)).detach().cpu().numpy()

		states_action_critic = np.zeros((ma_controller.num_agents,ma_controller.agents.value_input_dim))
		for i in range(ma_controller.num_agents):
			# other_actions_temp = np.delete(other_actions,i,axis=0)
			policies_temp = np.delete(policies,i,axis=0)
			tmp = np.copy(states_critic[i])
			for j in range(ma_controller.num_agents-1):
				if j == ma_controller.num_agents-2:
					# tmp = np.concatenate([tmp,other_actions_temp[j]])
					tmp = np.concatenate([tmp,policies_temp[j]])

					continue
				# tmp = np.concatenate([tmp[:-6*(ma_controller.num_agents-j-2)],other_actions_temp[j],tmp[-6*(ma_controller.num_agents-j-2):]])
				tmp = np.concatenate([tmp[:-6*(ma_controller.num_agents-j-2)],policies_temp[j],tmp[-6*(ma_controller.num_agents-j-2):]])
			states_action_critic[i] = tmp

		# Advance a step and render a new image
		with torch.no_grad():
			action = ma_controller.get_actions(states_actor)

		next_states,rewards,dones,info = env.step(action)

		next_states_critic,next_states_actor = ma_controller.split_states(next_states)

		img = env.render(mode='rgb_array')


	# print(images)

	# imageio.mimwrite('./simple_spread.gif',
	#                 [np.array(img) for i, img in enumerate(images) if i%2 == 0],
	#                 fps=50)

	# print("DONE!")

# import pyglet
# window = pyglet.window.Window()
# pyglet.app.run()
