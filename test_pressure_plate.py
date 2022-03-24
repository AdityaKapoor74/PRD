import gym
import numpy as np
import random
import pressureplate
import time

env = gym.make('pressureplate-linear-6p-v0')
for episode in range(100):
	obs = env.reset()
	# print(obs[0][:-2].reshape(5,5,5))
	terminated = False
	while not(terminated):
		action = [random.randint(0,4) for i in range(6)]
		# action = [4 for i in range(6)]
		obs, reward, terminated, info = env.step(action)
		# print("*"*10)
		# print(obs[0][:-2].reshape(5,5,5))
		env.render()
		time.sleep(10.0)
		# break
	# break