import sys
sys.path.insert(1, '../QLearning')

import os
os.chdir('C:/Users/nykim/Documents/ECE6254_ML/Project/ece6254_project/QLearning')
os.getcwd()

import util
import gym
import QLearningAgent
import trafficEnv2 as te

import matplotlib.pyplot as plt
import numpy as np

def normalizeState(obs, k): #observation, number of components for waits
    obs_wait = obs[-k:]
    obs_wait = obs_wait - min(obs_wait)
    obs[-k:] = obs_wait
    return obs

#env = gym.make('Taxi-v3')
speed = 2
for speed in range(3):
    env = te.TrafficEnv(car_speed=speed, max_wait=100, 
                        horiz_lanes=('e',), vert_lanes=('n',), 
                        horiz_sizes=(3,3), vert_sizes=(3,3))
    env.make_spawn_blocks(env.start_indices,[0.1,0.6])# [0.5 for _ in range(len(env.start_indices))])
    env_numLanes = (len(env.horiz_lanes) + len(env.vert_lanes))
    
    #initcars = normalizeState(env.observation_space.sample(), env_numLanes)
    #observation = env.reset(initcars)#env.observation_space.sample())
    if speed == 0:
        observation = env.reset(env.waitline_sizes)
    else:
        observation = env.reset([True for _ in range(len(env.valid_car_indices[0]))])
    env.render()
    observation
    
    QLearner = QLearningAgent.QLearningAgent(env.observation_space, env.action_space, 
                                             epsilon=0.8, alpha=0.2, discount=0.9)
    
    num_episodes = 5000 ## increase the episodes
    eposodes_len = 100
    nlight = len(env.action_space.nvec)
    discountedRwrd = []
    discountfactor = 0.9
    trainreward = []
    
    # fig1 = plt.figure()
    # ax = fig1.add_subplot(111)
    # line1, = ax.plot([], [])
    # line2, = ax.plot([], [], 'tab:orange')
    # ax.axis([0, num_episodes, 50, 100])
    # plt.pause(0.001)
    # background = fig1.canvas.copy_from_bbox(ax.bbox)
    
    for episode in range(0, num_episodes):
        #initcars = normalizeState(env.observation_space.sample(), env_numLanes)
        #observation = env.reset(initcars)
        if speed == 0:
            observation = env.reset(env.waitline_sizes)
        else:
            observation = env.reset([True for _ in range(len(env.valid_car_indices[0]))])
        discountedr=0
        QLearner.alpha = 0.8
        for j in range(eposodes_len):#while True:
            if j % (eposodes_len / 100) == 0:
                QLearner.alpha = 0.8*(0.7**(100*j/eposodes_len))
            action_num = QLearner.getAction(tuple(observation))
            action = [int(i) for i in bin(action_num)[2:]]
            for k in range(nlight-len(action)):
                action = [0]+action
            next_observation, reward, done, info = env.step(action)
            if reward >= 0:
                discountedr += reward*(discountfactor**j)
            QLearner.update(tuple(observation), action_num, tuple(next_observation), reward)
            observation = next_observation
            if reward < -1000:
                discountedRwrd = discountedRwrd + [discountedr]
                break
        discountedRwrd = discountedRwrd + [discountedr]
        if episode % (num_episodes / 100) == 0:
            util.printProgressBar(episode, num_episodes)
        # if episode % 100 == 0 or episode == num_episodes-1:
        #     fig1.canvas.restore_region(background)
        #     line1.set_xdata(range(0, len(discountedRwrd), 10))
        #     line1.set_ydata(discountedRwrd[0::10])
        #     line2.set_xdata(range(99, len(discountedRwrd), 10))
        #     line2.set_ydata([np.nanmean(discountedRwrd[(i+1-100):(i+1)])
        #         for i in range(99, len(discountedRwrd), 10)])
        #     ax.draw_artist(line1)
        #     ax.draw_artist(line2)
        #     fig1.canvas.blit(ax.bbox)
        #     plt.pause(0.001)
    print("DONE TRAINING")
    
    discountedRwrd_smth = [sum(discountedRwrd[max(0, i-100):i])/100 for i in range(len(discountedRwrd))]
    
    #for results in epreward_rand:
    QLtrain = plt.plot(discountedRwrd)
    QLtrain = plt.plot(discountedRwrd_smth, label='smoothed')
    QLtrain = plt.xlabel('Train Episodes with speed =%i'%(speed))
    QLtrain = plt.ylabel('Rewards')
    plt.show()
    
    
    
    numExperiment = 100
    QLearner.epsilon = 0.1
    QLearner.alpha =0.2
        
    epreward = []
    epreward_dscnt = []
    max_wait_reached = 0
    for episode in range(0, numExperiment):
        #initcars = normalizeState(env.observation_space.sample(), env_numLanes)
        #observation = env.reset(initcars)
        if speed == 0:
            observation = env.reset(env.waitline_sizes)
        else:
            observation = env.reset([True for _ in range(len(env.valid_car_indices[0]))])
        discountedr=0
        #env.render()
        r = []
        rd = 0
        for i in range(0, eposodes_len):
            action_num = QLearner.getAction(tuple(observation), False)
            action = [int(i) for i in bin(action_num)[2:]]
            for k in range(nlight-len(action)):
                action = [0]+action
            next_observation, reward, done, info = env.step(action)
            #env.render()
            #print(reward)
            r = r + [reward]
            rd += reward*(discountfactor**i)
            observation = next_observation
            if reward < -100:
                max_wait_reached+=1
                break
        epreward = epreward + [r]
        epreward_dscnt = epreward_dscnt+ [rd]
    
    env.step(action)
    env.render()
    
    #for results in epreward:
    for k in range(0,5):
         resultsovertime = plt.plot(epreward[k])
         resultsovertime = plt.xlabel('Steps')
         resultsovertime = plt.ylabel('Rewards')
         resultsovertime = plt.title('Q-learner rewards, speed=%i'%(speed))
    plt.show()
    
    ##Let's compare with random actions
    epreward_rand = []
    epreward_rand_dscnt = []
    max_wait_reached_rand=0
    for episode in range(0, numExperiment):
        #initcars = normalizeState(env.observation_space.sample(), env_numLanes)
        #observation = env.reset(initcars)
        if speed == 0:
            observation = env.reset(env.waitline_sizes)
        else:
            observation = env.reset([True for _ in range(len(env.valid_car_indices[0]))])
        discountedr=0
        #env.render()
        r = []
        rd=0
        for i in range(0, eposodes_len):
            action = env.action_space.sample()
            next_observation, reward, done, info = env.step(action)
            r = r + [reward]
            rd += reward*(discountfactor**i)
            observation = next_observation
            if reward < -100:
                max_wait_reached_rand +=1
                break
            # else:
            #     r = r + [reward]
            #     rd += reward*(discountfactor**i)
            #     observation = next_observation
        epreward_rand = epreward_rand + [r]
        epreward_rand_dscnt = epreward_rand_dscnt + [rd]
    
    #for results in epreward_rand:
    for k in range(0,5):
         resultsovertime = plt.plot(epreward_rand[k])
         resultsovertime = plt.xlabel('Steps')
         resultsovertime = plt.ylabel('Rewards')
         resultsovertime = plt.title('Random-action rewards, speed=%i'%(speed))
    plt.show()
    
    print('speed = %i'%(speed))
    print(sum([sum(i) for i in epreward]))
    print(sum([sum(i) for i in epreward_rand]))
    print(sum([sum(i) for i in epreward])/sum([sum(i) for i in epreward_rand]))
    
    print(sum(epreward_dscnt))
    print(sum(epreward_rand_dscnt))
    print(sum(epreward_dscnt)/sum(epreward_rand_dscnt))
    
    print(max_wait_reached)
    print(max_wait_reached_rand)


#env.close()