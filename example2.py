import sys
sys.path.insert(1, '../QLearning')

import util
import gym
import QLearningAgent
import trafficEnv2 as te

#env = gym.make('Taxi-v3')
env = te.TrafficEnv(car_speed=0)
env.make_spawn_blocks(env.start_indices, [0.5 for _ in range(len(env.start_indices))])

QLearner = QLearningAgent.QLearningAgent(env.observation_space, env.action_space, epsilon=0.3, alpha=0.2, discount=0.9)
#env.step([[0],[0],[0],[0],[0],[0]])
num_episodes = 1
nlight = len(env.action_space.nvec)
r = []
for episode in range(0, num_episodes):
    observation = env.reset(env.observation_space.sample())
    while True:
        action_num = QLearner.getAction(tuple(observation))
        action = [[int(i)] for i in bin(action_num)[2:]]
        for k in range(nlight-len(action)):
            action = [[0]]+action
        next_observation, reward, done, info = env.step(action)
        print(reward)
        r = r + [reward]
        QLearner.update(tuple(observation), action_num, tuple(next_observation), reward)
        observation = next_observation
        if done:
            break
    
    if episode % (num_episodes / 100) == 0:
        util.printProgressBar(episode, num_episodes)

print("DONE TRAINING")

for episode in range(0, 1):
    observation = env.reset(env.observation_space.sample())
    env.render()
    for i in range(0, 10):
        action_num = QLearner.getAction(tuple(observation), False)
        action = [[int(i)] for i in bin(action_num)[2:]]
        for k in range(nlight-len(action)):
            action = [[0]]+action
        next_observation, reward, done, info = env.step(action)
        env.render()
        print(reward)
        observation = next_observation
        if done:
            print("Success!")
            print("")
            print("-------------------------------------------")
            print("")
            break

env.close()