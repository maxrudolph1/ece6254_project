import sys
sys.path.insert(1, '../QLearning')

import os
os.chdir('C:/Users/nykim/Documents/ECE6254_ML/Project/ece6254_project/QLearning')
os.getcwd()

import util
import gym
import QLearningAgent
import trafficEnv2 as te

def normalizeState(obs, k): #observation, number of components for waits
    obs_wait = obs[-k:]
    obs_wait = obs_wait - min(obs_wait)
    obs[-k:] = obs_wait
    return obs

#env = gym.make('Taxi-v3')
env = te.TrafficEnv(car_speed=0, max_wait=1000)
env.make_spawn_blocks(env.start_indices, [0.5 for _ in range(len(env.start_indices))])
env_numLanes = (len(env.horiz_lanes) + len(env.vert_lanes))

initcars = normalizeState(env.observation_space.sample(), env_numLanes)
observation = env.reset(initcars)#env.observation_space.sample())
env.render()
observation

# initcars[:-(len(env.horiz_lanes) + len(env.vert_lanes))]
# env.car_indices[env.valid_car_indices[0]]
# [2 for _ in range(len(self.valid_car_indices))]
#                 + [max_wait + 1 for _ in range(len(self.horiz_lanes) + len(self.vert_lanes))])

# ss = spaces.MultiDiscrete(
#                 [2 for _ in range(len(env.valid_car_indices))]
#                 + [100 + 1 for _ in range(len(env.horiz_lanes) + len(env.vert_lanes))])



QLearner = QLearningAgent.QLearningAgent(env.observation_space, env.action_space, epsilon=0.3, alpha=0.2, discount=0.9)

num_episodes = 1000 ## increase the episodes
eposodes_len = 100
nlight = len(env.action_space.nvec)
#r = []
discountedRwrd = []
discountfactor = 0.9
for episode in range(0, num_episodes):
    initcars = normalizeState(env.observation_space.sample(), env_numLanes)
    observation = env.reset(initcars)
    discountedr=0
    for j in range(eposodes_len):#while True:
        action_num = QLearner.getAction(tuple(observation))
        action = [int(i) for i in bin(action_num)[2:]]
        for k in range(nlight-len(action)):
            action = [0]+action
        next_observation, reward, done, info = env.step(action)
        #(reward)
        #r = r + [reward]
        if reward >= 0:
            discountedr += reward*(discountfactor**j)
        QLearner.update(tuple(observation), action_num, tuple(next_observation), reward)
        observation = next_observation
        # if done:
        #     break
        if reward < -1000:
            discountedRwrd = discountedRwrd + [discountedr]
            break
    discountedRwrd = discountedRwrd + [discountedr]
    if episode % (num_episodes / 100) == 0:
        util.printProgressBar(episode, num_episodes)
print("DONE TRAINING")

epreward = []
for episode in range(0, 5):
    observation = env.reset(env.observation_space.sample())
    env.render()
    r = []
    for i in range(0, eposodes_len):
        action_num = QLearner.getAction(tuple(observation), False)
        action = [int(i) for i in bin(action_num)[2:]]
        for k in range(nlight-len(action)):
            action = [0]+action
        next_observation, reward, done, info = env.step(action)
        #env.render()
        #print(reward)
        r = r + [reward]
        observation = next_observation
        if reward < -100:
            break
    epreward = epreward + [r]
        # if done:
        #     print("Success!")
        #     print("")
        #     print("-------------------------------------------")
        #     print("")
        #     break

env.step(action)
env.render()

import matplotlib.pyplot as plt
for results in epreward:
     resultsovertime = plt.plot(results)
     resultsovertime = plt.xlabel('Steps')
     resultsovertime = plt.ylabel('Rewards')
plt.show()

##Let's compare with random actions
env.observation_space.sample()
epreward_rand = []
for episode in range(0, 5):
    observation = env.reset(env.observation_space.sample())
    env.render()
    r = []
    for i in range(0, eposodes_len):
        action = env.action_space.sample()
        next_observation, reward, done, info = env.step(action)
        #env.render()
        #print(reward)
        r = r + [reward]
        observation = next_observation
        if reward < -100:
            break
    epreward_rand = epreward_rand + [r]

for results in epreward_rand:
     resultsovertime = plt.plot(results)
     resultsovertime = plt.xlabel('Steps')
     resultsovertime = plt.ylabel('Rewards')
plt.show()

plt_discReward = plt.plot(discRewards_Rand)
plt.show()

discRewards_Q = []
for r in epreward:
    rw = 0
    for i in range(len(r)):
        rw += r[i]*(discountfactor**i)
    discRewards_Q=discRewards_Q +[rw]
    
discRewards_Rand = []
for r in epreward_rand:
    rw = 0
    for i in range(len(r)):
        rw += r[i]*(discountfactor**i)
    discRewards_Rand=discRewards_Rand +[rw]


env.close()