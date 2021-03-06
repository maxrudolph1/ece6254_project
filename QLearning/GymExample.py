import util
import gym
import QLearningAgent

env = gym.make('Taxi-v3')
QLearner = QLearningAgent.QLearningAgent(env.observation_space, env.action_space, epsilon=0.3, alpha=0.2, discount=0.9)

num_episodes = 1000
for episode in range(0, num_episodes):
    observation = env.reset()
    while True:
        action = QLearner.getAction(observation)
        next_observation, reward, done, info = env.step(action)
        QLearner.update(observation, action, next_observation, reward)
        observation = next_observation
        if done:
            break
    
    if episode % (num_episodes / 100) == 0:
        util.printProgressBar(episode, num_episodes)

print("DONE TRAINING")

for episode in range(0, 4):
    observation = env.reset()
    env.render()
    for i in range(0, 1000):
        action = QLearner.getAction(observation, False)
        next_observation, reward, done, info = env.step(action)
        env.render()
        observation = next_observation
        if done:
            print("Success!")
            print("")
            print("-------------------------------------------")
            print("")
            break

env.close()
