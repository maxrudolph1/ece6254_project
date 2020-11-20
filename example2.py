import sys
sys.path.insert(1, 'QLearning')
import matplotlib.pyplot as plt
import numpy as np
import util
import gym
import QLearningAgent
import trafficEnv2 as te
    

num_episodes = 1000 ## increase the episodes
eposodes_len = 100
discountfactor = 0.9


#env = te.TrafficEnv(car_speed=0, max_wait=1000)
#env.make_spawn_blocks(env.start_indices, [0.1, 0.1, 0.6, 0.1, 0.6, 0.1])

env = te.TrafficEnv(car_speed=0, max_wait=100, horiz_lanes=('e',), vert_lanes=('n',), horiz_sizes=(7, 7), vert_sizes=(3, 3))
env.make_spawn_blocks(env.start_indices, [0.1, 0.6])


QLearner = QLearningAgent.QLearningAgent(env.observation_space, env.action_space, 
    epsilon=0.05*1+0, alpha=0.2, discount=discountfactor)

nlight = len(env.action_space.nvec)
#r = []
discountedRwrd = []


fig1 = plt.figure()
ax = fig1.add_subplot(111)
line1, = ax.plot([], [])
line2, = ax.plot([], [], 'tab:orange')
ax.axis([0, num_episodes, 50, 100])
plt.pause(0.001)
background = fig1.canvas.copy_from_bbox(ax.bbox)

for episode in range(num_episodes):
    if env.has_inf_speed:
        observation = env.reset(env.waitline_sizes)
    else:
        observation = env.reset([True for _ in range(len(env.valid_car_indices[0]))])
    discountedr = 0
    for j in range(eposodes_len):#while True:
        if episode == num_episodes-1:
            env.render()
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
        if reward < 0:
            break
    discountedRwrd = discountedRwrd + [discountedr]
    if reward < 0:
        discountedRwrd[-1] = np.nan
    
    if episode % 100 == 0 or episode == num_episodes-1:
        fig1.canvas.restore_region(background)
        line1.set_xdata(range(0, len(discountedRwrd), 10))
        line1.set_ydata(discountedRwrd[0::10])
        line2.set_xdata(range(99, len(discountedRwrd), 10))
        line2.set_ydata([np.nanmean(discountedRwrd[(i+1-100):(i+1)])
            for i in range(99, len(discountedRwrd), 10)])
        ax.draw_artist(line1)
        ax.draw_artist(line2)
        fig1.canvas.blit(ax.bbox)
        plt.pause(0.001)
    
    if episode % (num_episodes / 100) == 0:
        util.printProgressBar(episode, num_episodes)
print("DONE TRAINING")
print(len(QLearner.q_values))


# epreward = []
# for episode in range(0, 5):
    # observation = env.reset(env.observation_space.sample())
    # env.render()
    # r = []
    # for i in range(0, eposodes_len):
        # action_num = QLearner.getAction(tuple(observation), False)
        # action = [int(i) for i in bin(action_num)[2:]]
        # for k in range(nlight-len(action)):
            # action = [0]+action
        # next_observation, reward, done, info = env.step(action)
        # #env.render()
        # #print(reward)
        # r = r + [reward]
        # observation = next_observation
        # if reward < -100:
            # break
    # epreward = epreward + [r]
        # # if done:
        # #     print("Success!")
        # #     print("")
        # #     print("-------------------------------------------")
        # #     print("")
        # #     break

# env.step(action)
# env.render()


# for results in epreward:
     # resultsovertime = plt.plot(results)
     # resultsovertime = plt.xlabel('Steps')
     # resultsovertime = plt.ylabel('Rewards')
# plt.show()

# ##Let's compare with random actions
# env.observation_space.sample()
# epreward_rand = []
# for episode in range(0, 5):
    # observation = env.reset(env.observation_space.sample())
    # env.render()
    # r = []
    # for i in range(0, eposodes_len):
        # action = env.action_space.sample()
        # next_observation, reward, done, info = env.step(action)
        # #env.render()
        # #print(reward)
        # r = r + [reward]
        # observation = next_observation
        # if reward < -100:
            # break
    # epreward_rand = epreward_rand + [r]

# for results in epreward_rand:
     # resultsovertime = plt.plot(results)
     # resultsovertime = plt.xlabel('Steps')
     # resultsovertime = plt.ylabel('Rewards')
# plt.show()

# plt_discReward = plt.plot(discRewards_Rand)
# plt.show()

# discRewards_Q = []
# for r in epreward:
    # rw = 0
    # for i in range(len(r)):
        # rw += r[i]*(discountfactor**i)
    # discRewards_Q=discRewards_Q +[rw]
    
# discRewards_Rand = []
# for r in epreward_rand:
    # rw = 0
    # for i in range(len(r)):
        # rw += r[i]*(discountfactor**i)
    # discRewards_Rand=discRewards_Rand +[rw]
    
plt.show(block=True)
