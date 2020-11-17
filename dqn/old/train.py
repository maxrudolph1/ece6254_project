import gym

import sys
from model import DQN
sys.path.insert(0,'../..')
from TrafficEnvironment import traffic_environment as te


import numpy as np
import torch

env = te.TrafficEnv(horiz_lanes=('e','e'), vert_lanes=('n','s','n'), 
horiz_sizes=(3,3,3,3), vert_sizes=(3,3,3), car_speed=3, max_steps=10)
obs_shape = torch.tensor(env.observation()).shape[0]
act_shape = env.action_space.shape[0]
print(obs_shape)
print(act_shape)
model = DQN(6, 6)



done = False
while not done:
    action = (np.random.random((6, 1)) > 0.5) + 0
    observation, reward, done, _ = env.step(action)
    model(torch.tensor(observation, dtype=torch.float))