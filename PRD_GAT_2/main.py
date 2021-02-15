# from maa2c import MAA2C
from maa2c import MAA2C

from multiagent.environment import MultiAgentEnv
# from multiagent.scenarios.simple_spread import Scenario
import multiagent.scenarios as scenarios
import torch 
import numpy as np

def make_env(scenario_name, benchmark=False):
	# load scenario from script
	scenario = scenarios.load(scenario_name + ".py").Scenario()
	# scenario = Scenario()
	# create world
	world = scenario.make_world()
	# create multiagent environment
	if benchmark:
		env = MultiAgentEnv(world, scenario.reset_world, scenario.reward_by_proximity, scenario.observation, scenario.benchmark_data, scenario.isFinished)
	else:
		env = MultiAgentEnv(world, scenario.reset_world, scenario.reward_by_proximity, scenario.observation, None, scenario.isFinished)
	return env



if __name__ == '__main__':
	env = make_env(scenario_name="paired_by_proximity",benchmark=False)
	ma_controller = MAA2C(env,gif=False,save=True)

	ma_controller.run(100000,100)
