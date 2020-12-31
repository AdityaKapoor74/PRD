from maa2c import MAA2C
import imageio
import numpy as np
import os
import torch

from multiagent.environment import MultiAgentEnv
# from multiagent.scenarios.simple_spread import Scenario
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
	n_images = 10000

	images = []

	# init a new episode
	obs = env.reset()
	# init the img var with the starting state of the env
	img = env.render(mode='rgb_array')[0]

	for i in range(n_images):
	  # At each step, append an image to list
	  images.append(img)

	  # Advance a step and render a new image
	  with torch.no_grad():
	    action = ma_controller.get_actions(obs)
	  obs, _, _ ,_ = env.step(action)
	  img = env.render(mode='rgb_array')[0]


	# print(images)

	# imageio.mimwrite('./simple_spread.gif',
	#                 [np.array(img) for i, img in enumerate(images) if i%2 == 0],
	#                 fps=50)

	# print("DONE!")

# import pyglet
# window = pyglet.window.Window()
# pyglet.app.run()
