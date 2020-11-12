import gym

import sys
from dqn import Net
sys.path.insert(0,'..')
from TrafficEnvironment import traffic_environment as te
import numpy as np
import torch

env = te.TrafficEnv(horiz_lanes=('e','e'), vert_lanes=('n','s','n'), 
horiz_sizes=(3,3,3,3), vert_sizes=(3,3,3), car_speed=2, max_steps=1000)

model = Net(6, 6)


env.render()
done = False
while not done:
    action = np.zeros((6, 1))
    observation, reward, done, _ = env.step(action)
    print(torch.Tensor(action).shape)
    env.render()
    done = True