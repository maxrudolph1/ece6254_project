import gym

import sys
from linear_qnet import Net
sys.path.insert(0,'..')
from TrafficEnvironment import traffic_environment as te


import numpy as np
import torch

env = te.TrafficEnv(horiz_lanes=('e','e'), vert_lanes=('n','s','n'), 
horiz_sizes=(3,3,3,3), vert_sizes=(3,3,3), car_speed=2, max_steps=1000)

model = Net(env.observation_space.shape[0], env.action_space.shape[0])


env.render()
done = False
while not done:
    action = np.zeros((6, 1))
    observation, reward, done, _ = env.step(action)
    obs = np.zeros(env.observation_space.shape)
    obs[observation] = 1

    print(model(torch.Tensor(obs)))
    env.render()
    done = True