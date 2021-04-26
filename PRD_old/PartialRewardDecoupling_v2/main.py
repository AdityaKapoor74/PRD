# from maa2c import MAA2C
from maa2c_revised import MAA2C

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
        env = MultiAgentEnv(world, scenario.reset_world, scenario.reward_agent_centric_paired, scenario.observation, scenario.benchmark_data, scenario.isFinished)
    else:
        env = MultiAgentEnv(world, scenario.reset_world, scenario.reward_agent_centric_paired, scenario.observation, None, scenario.isFinished)
    return env



if __name__ == '__main__':
	env = make_env(scenario_name="custom_env",benchmark=False)
	ma_controller = MAA2C(env,False)

	ma_controller.run(10000000,300)
