import gym

import sys
from linear_qnet import Net
sys.path.insert(0,'..')
from TrafficEnvironment import traffic_environment as te


import numpy as np
import torch

env = te.TrafficEnv(horiz_lanes=('e','e', 'e'), vert_lanes=('n','n','s'), 
horiz_sizes=(3,3,3,3), vert_sizes=(3,3,3,3), car_speed=2, max_steps=1000)

model = Net(env.observation_space.shape[0], env.action_space.shape[0])


print(env.observation_space.shape)
print(env.action_space)
env.render()
done = False
count = 0
while not done:
    action = np.zeros((env.action_space.shape))
    action[np.random.random((env.action_space.shape)) > 0] = 1
    observation, reward, done, _ = env.step(action)
    count += 1
    if count > 2:
        break
    env.render()



env.render()
    