import gym
import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple
from itertools import count
from PIL import Image
from model import DQN

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
#from replay_buffer import ReplayMemory
import sys
sys.path.insert(0,'..')
from TrafficEnvironment import traffic_environment as te

# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

plt.ion()

env = te.TrafficEnv(horiz_lanes=('e','w', 'e'), vert_lanes=('n','s', 's'), horiz_sizes=(10,10, 10,10), vert_sizes=(10,10,10,10), 
                    car_speed=2, max_wait=100, max_wait_penalty=1000, max_steps=200)


# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


BATCH_SIZE = 32
GAMMA = 0.999
EPS_START = .99
EPS_END = 0.05
EPS_DECAY = 200
TARGET_UPDATE = 10


Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
# Get screen size so that we can initialize layers correctly based on shape
# returned from AI gym. Typical dimensions at this point are close to 3x40x90
# which is the result of a clamped and down-scaled render buffer in get_screen()


# Get number of actions from gym action space


n_actions = 2**env.action_space.shape[0]
state_space = torch.tensor(env.observation()).shape[0]

policy_net = DQN(state_space, n_actions).to(device)
target_net = DQN(state_space, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.RMSprop(policy_net.parameters())
memory = ReplayMemory(10000)


steps_done = 0

def converter(state):
    return torch.tensor(state).float().unsqueeze(0)

def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    state = converter(state)
    action = torch.zeros((env.action_space.shape))

    if sample > eps_threshold:
        with torch.no_grad():
            # t.max(1) will return largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            num_lights = env.action_space.shape[0]
            val = policy_net(state).argmax(dim=1)
            action = torch.tensor([int(x) for x in bin(val)[2:]]).float()
            action = torch.cat((torch.zeros((num_lights)), action), dim=0)
            action = action[-num_lights:]
            
            return action
    else:

        return (torch.rand(env.action_space.shape[0]) > 0.5) + 0


episode_durations = []


def plot_durations():
    plt.figure(2)
    plt.clf()
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    plt.title('Training...')
    plt.xlabel('Episodes')
    plt.ylabel('Reward')
    
    plt.plot(durations_t.numpy())
    # Take 100 episode averages and plot them too
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 10, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(9), means))
        plt.plot(means.numpy())

    plt.pause(0.001)  # pause a bit so that plots are updated
    if is_ipython:
        display.clear_output(wait=True)
        display.display(plt.gcf())


def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net


    state_action_values = policy_net(state_batch).gather(1, action_batch.long())

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1)[0].
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()


num_episodes = 500

state = env.observation()

first_action = select_action(state)
first_state = state


for i_episode in range(num_episodes):
    # Initialize the environment and state
    env.reset([True for _ in range(len(env.valid_car_indices))])

    state = env.observation()
    expert_policy = torch.tensor([0,0,0,0])
    episode_reward = 0
    for t in count():
        # Select and perform an action

        
        action = select_action(state)

        # if t % 4 == 0:
        #     #print('here')
        #     expert_policy = 1 - expert_policy
 
        
        next_state, reward, done, _ = env.step(action)
        reward = torch.tensor([reward], device=device)

        # Observe new state

        arr = torch.pow(2, torch.linspace(env.action_space.shape[0]-1,0, env.action_space.shape[0]))
        
        action_idx = (torch.sum(arr*action)).unsqueeze(-1).unsqueeze(-1)
        
        # Store the transition in memory
        memory.push(converter(state), action_idx, converter(next_state), reward)

        # Move to the next state
        state = next_state

        # Perform one step of the optimization (on the target network)
        optimize_model()
        
        episode_reward += reward*(GAMMA**t)

        if done:
            episode_durations.append(episode_reward)
            plot_durations()
            break

    # Update the target network, copying all weights and biases in DQN
    if i_episode % TARGET_UPDATE == 0:
        target_net.load_state_dict(policy_net.state_dict())

print('Complete')
env.render()
plt.ioff()
plt.show()