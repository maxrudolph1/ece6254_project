from gym.utils import seeding
import numpy as np
#import math
#import time
##main git commands for reference: clone, checkout, branch, add, commit, push, status

class TrafficEnv():
    def __init__(self, horiz_lanes=('ew',), vert_lanes=('sn',), horiz_sizes=(10,10), vert_sizes=(10,10), 
                    car_speed=2, max_steps=1000, arrival_rate=[0.5, 0.5,0.5,0.5], 
                    max_wait=500, max_wait_penalty=100000):
        self.horiz_lanes = horiz_lanes
        self.vert_lanes = vert_lanes
        self.horiz_sizes = horiz_sizes
        self.vert_sizes = vert_sizes
        self.car_speed = car_speed
        self.max_steps = max_steps
        self.arrRate = arrival_rate # start from horizontal lanes 
        self.max_wait=max_wait
        self.max_wait_penalty = max_wait_penalty 
        self.initial_cars = None
        self.current_step = 0
        self.verify_inputs()
        self.get_layout()
        # Make all lights green for vertical lanes ('1'), it does not matter
        self.lights = [0 for _ in range(len(self.horiz_lanes) * len(self.vert_lanes))]
        self.make_initial_state([True for _ in range(self.ly * self.lx)])
        self.reset()
        
    
    def verify_inputs(self):
        # Verify 'horiz_lanes'
        if type(self.horiz_lanes) != tuple:
            raise TypeError('Argument \'horiz_lanes\' must be a tuple of strings')
        elif len(self.horiz_lanes) <= 0:
            raise ValueError('Argument \'horiz_lanes\' is empty')
        else:
            for k in range(len(self.horiz_lanes)):
                if type(self.horiz_lanes[k]) != str:
                    raise TypeError('Entry %d of argument \'horiz_lanes\' is not a string' % k)
                elif len(self.horiz_lanes[k]) <= 0:
                    raise ValueError('Entry %d of argument \'horiz_lanes\' is an empty string' % k)
                else:
                    for i in range(len(self.horiz_lanes[k])):
                        if not set(self.horiz_lanes[k][i]).issubset(set('eEwW')):
                            raise ValueError('Entry %d of argument \'horiz_lanes\' has an invalid character' % k)
        # Verify 'vert_lanes'
        if type(self.vert_lanes) != tuple:
            raise TypeError('Argument \'vert_lanes\' must be a tuple of strings')
        elif len(self.vert_lanes) <= 0:
            raise ValueError('Argument \'vert_lanes\' is empty')
        else:
            for k in range(len(self.vert_lanes)):
                if type(self.vert_lanes[k]) != str:
                    raise TypeError('Entry %d of argument \'vert_lanes\' is not a string' % k)
                elif len(self.vert_lanes[k]) <= 0:
                    raise ValueError('Entry %d of argument \'vert_lanes\' is an empty string' % k)
                else:
                    for i in range(len(self.vert_lanes[k])):
                        if not set(self.vert_lanes[k][i]).issubset(set('nNsS')):
                            raise ValueError('Entry %d of argument \'vert_lanes\' has an invalid character' % k)
        # Verify 'horiz_sizes'
        if type(self.horiz_sizes) != tuple:
            raise TypeError('Argument \'horiz_sizes\' must be a tuple of positive ints')
        elif len(self.horiz_sizes) != len(self.vert_lanes) + 1:
            raise TypeError('Argument \'horiz_sizes\' must have the length of argument \'vert_lanes\' plus 1')
        for k in range(len(self.horiz_sizes)):
            if type(self.horiz_sizes[k]) != int:
                raise TypeError('Entry %d of argument \'horiz_sizes\' is not an int' % k)
            elif self.horiz_sizes[k] <= 0:
                raise ValueError('Entry %d of argument \'horiz_sizes\' is not positive' % k)
        # Verify 'vert_sizes'
        if type(self.vert_sizes) != tuple:
            raise TypeError('Argument \'vert_sizes\' must be a tuple of positive ints')
        elif len(self.vert_sizes) != len(self.horiz_lanes) + 1:
            raise TypeError('Argument \'vert_sizes\' must have the length of argument \'horiz_lanes\' plus 1')
        for k in range(len(self.vert_sizes)):
            if type(self.vert_sizes[k]) != int:
                raise TypeError('Entry %d of argument \'vert_sizes\' is not an int' % k)
            elif self.vert_sizes[k] <= 0:
                raise ValueError('Entry %d of argument \'vert_sizes\' is not positive' % k)
        # Verify 'car_speed'
        if type(self.car_speed) != int:
            raise TypeError('Argument \'car_speed\' must be a positive int')
        elif self.car_speed <= 0:
            raise ValueError('Argument \'car_speed\' is not positive')
        # Verify 'max_steps'
        if type(self.max_steps) != int:
            raise TypeError('Argument \'max_steps\' must be a positive int')
        elif self.max_steps <= 0:
            raise ValueError('Argument \'max_steps\' is not positive')

    
    def reward(self):
        # reward = 0.0
        # for lane in self.lanes:
        #    reward -= traci.lane.getWaitingTime(lane)
        # return reward
        speed = traci.multientryexit.getLastStepMeanSpeed(self.detector)
        count = traci.multientryexit.getLastStepVehicleNumber(self.detector)
        reward = speed * count
        # print("Speed: {}".format(traci.multientryexit.getLastStepMeanSpeed(self.detector)))
        # print("Count: {}".format(traci.multientryexit.getLastStepVehicleNumber(self.detector)))
        # reward = np.sqrt(speed)
        # print "Reward: {}".format(reward)
        # return speed
        # reward = 0.0
        # for loop in self.exitloops:
        #    reward += traci.inductionloop.getLastStepVehicleNumber(loop)
        return max(reward, 0)

    
    def observation(self):
        return self.car_indices.tolist()
        
    def move_car(self, ind):
    # TODO: explain outputs
        # Get car direction
        ind_step = self.layout_steps[ind]
        # Get next block
        ind_next = ind + ind_step
        block_next = self.layout[ind]
        if block_next < 0:
            # Car is entering an intersection
            if self.light_dict[ind_step] == self.lights[-block_next-1]:
                # Green light, get next block after intersection
                while self.layout[ind_next] == block_next:
                    ind_next += ind_step
            else:
                # Red light, car cannot move
                return False, ind
        if self.layout[ind_next] == 0:
            # Car reached a goal block
            self.car_indices[ind] = False
            self.car_waits[ind] = 0
            return False, None
        elif self.car_indices[ind_next]:
            # Next block is occupied
            return False, ind
        else:
            # Car can move to next block
            self.car_indices[ind] = False
            self.car_indices[ind_next] = True
            self.car_waits[ind_next] = self.car_waits[ind]+self.car_speed
            self.car_waits[ind] = 0
            return True, ind_next
    
    def enter_car(self):
        # Generate new cars
        for s in self.spawn_indices:
            ind_step = self.layout_steps[s]
            p = self.arrRate[self.spawn_indices.index(s)]
            for k in range(self.car_speed):
                if self.car_indices[s+k*ind_step] == True:
                    break
                else:
                    self.car_indices[s+k*ind_step] = (np.random.binomial(1,p)==1)       
        return True
    

    def get_reward(self):
        # reward = sum (wait_time if wait<max_wait, max_penalty OW)
        return -sum([x*x if x<self.max_wait else self.max_wait_penalty for x in self.car_waits])

    def step(self, action):
        if self.current_step < self.max_steps:
            # We save some overhead by not verifying input action
            self.lights = action
            # Make stack of cars by going through lanes in reverse
            car_stack = []
            for spawn_k in self.spawn_indices:
                ind = spawn_k
                ind_step = self.layout[ind] - ind
                while self.layout[ind] != 0:
                    if self.car_indices[ind]:
                        car_stack.append(ind)
                    ind += ind_step
            # Move cars until stack is empty
            len_stack = len(car_stack)
            while len_stack > 0:
                ind = car_stack.pop()
                len_stack -= 1
                for _ in range(self.car_speed):
                    can_move, ind = self.move_car(ind)
                    if not can_move:
                        break
            # Let new cars join
            self.enter_car()
            self.current_step += 1
        # Compute output variables
        observation = None
        reward = None
        done = self.current_step >= self.max_steps
        return observation, reward, done, None
    
    def get_layout(self):
        # TODO: Complete documentation
        """
        Layout convention:
        0:                  non-lane block
        positive number:    index of next block in path
        negative number:    index of intersection associated to next block (1-indexed)
        
        When a car goes to a non-lane block, it is removed from the layout
        """
        # Create layout matrix
        self.ly = sum(self.vert_sizes) + sum(len(str_k) for str_k in self.horiz_lanes)
        self.lx = sum(self.horiz_sizes) + sum(len(str_k) for str_k in self.vert_lanes)
        self.layout = np.zeros((self.ly, self.lx), dtype=np.int32)
        # Create layout step matrix, for moving cars
        self.layout_steps = np.zeros((self.ly, self.lx), dtype=np.int32)
        # List of spawn blocks
        self.spawn_indices = []
        # List of goal blocks
        self.goal_indices = []
        # List of lists
        # The i-th list has the indices of the blocks of the i-th intersection
        self.inter_indices = [[] 
            for _ in range(len(self.horiz_lanes) * len(self.vert_lanes))]
        # Get directions of horizontal lane blocks
        ind = 0
        for k in range(len(self.horiz_lanes)):
            ind += self.vert_sizes[k]
            for c in self.horiz_lanes[k]:
                if c == 'e' or c == 'E':
                    self.layout[ind,:] = ind + self.ly * (np.arange(0, self.lx) + 1)
                    self.layout[ind,-1] = 0
                    self.layout_steps[ind,:] = 1
                    self.spawn_indices.append(ind)
                    self.goal_indices.append(ind + self.ly * (self.lx - 1))
                else:
                    self.layout[ind,:] = ind + self.ly * (np.arange(0, self.lx) - 1)
                    self.layout[ind,0] = 0
                    self.layout_steps[ind,:] = 3
                    self.spawn_indices.append(ind + self.ly * (self.lx - 1))
                    self.goal_indices.append(ind)
                ind += 1
        # Get directions of vertical lane blocks and intersections
        ind = 0
        for k in range(len(self.vert_lanes)):
            ind += self.horiz_sizes[k]
            for c in self.vert_lanes[k]:
                # Fill vertical lane blocks
                if c == 'n' or c == 'N':
                    self.layout[:,ind] = self.ly * ind + (np.arange(0, self.ly, step=1) - 1)
                    self.layout[0,ind] = 0
                    self.layout_steps[:,ind] = 2
                    self.spawn_indices.append(self.ly * (ind + 1) - 1)
                    self.goal_indices.append(self.ly * ind)
                else:
                    self.layout[:,ind] = self.ly * ind + (np.arange(0, self.ly) + 1)
                    self.layout[-1,ind] = 0
                    self.layout_steps[:,ind] = 4
                    self.spawn_indices.append(self.ly * ind)
                    self.goal_indices.append(self.ly * (ind + 1) - 1)
                # Fill intersections for this vertical lane
                ind_row = 0
                for i in range(len(self.horiz_lanes)):
                    ind_row += self.vert_sizes[i]
                    inter_j = i + k * len(self.horiz_lanes)
                    ind_inter = np.arange(ind_row, ind_row + len(self.horiz_lanes[i]), step=1)
                    self.layout[ind_inter,ind] = -(inter_j + 1)
                    self.inter_indices[inter_j].extend((ind_inter + ind * self.ly).tolist())
                    ind_row += len(self.horiz_lanes[i])
                ind += 1
        # Convert layout matrix to vector by concatenating columns
        self.layout = np.ravel(self.layout, order='F')
        # Find blocks that point to an intersection block
        ind_inter = np.logical_and(self.layout > 0, self.layout[self.layout] < 0)
        # And change their value for the intersection index instead
        self.layout[ind_inter] = self.layout[self.layout[ind_inter]]
        # Convert layout step matrix to vector by concatenating columns
        self.layout_steps = np.ravel(self.layout_steps, order='F')
        # Compute the step size of every direction
        east_blocks = self.layout_steps == 1
        north_blocks = self.layout_steps == 2
        west_blocks = self.layout_steps == 3
        south_blocks = self.layout_steps == 4
        self.layout_steps[east_blocks] = self.ly
        self.layout_steps[north_blocks] = -1
        self.layout_steps[west_blocks] = -self.ly
        self.layout_steps[south_blocks] = 1
        # Car character dictionary, for rendering purposes
        self.car_dict = {
            self.ly: '>',
            -1: '^',
            -self.ly: '<',
            1: 'v'
        }
        # Light dictionary, gives the required light value to pass through
        # an intersection in the given direction
        self.light_dict = {
            self.ly: 0,
            -1: 1,
            -self.ly: 0,
            1: 1
        }
        
        
    def make_initial_state(self, cars):
        # This method must be called before using the object!
        # Verify input
        if type(cars) != tuple and type(cars) != list and type(cars) != np.ndarray:
            raise TypeError('Argument \'cars\' must be a tuple or list or numpy.ndarray')
        elif len(cars) <= 0:
            raise ValueError('Argument \'cars\' is empty')
        else:
            for k in range(len(cars)):
                if type(cars[k]) != bool:
                    raise TypeError('Entry %d of argument \'cars\' is not a bool' % k)
        self.initial_cars = np.array(cars)
    
    
    def reset(self, seed=None):
        # Reseed random generator
        self.np_random, seed = seeding.np_random(seed)
        # Reset cars
        # Delete cars in non-lane blocks
        self.car_indices = np.array(self.initial_cars)
        self.car_indices[self.layout[self.car_indices] == 0] = False
        # Delete cars in intersection blocks
        inter_list = [inner for outer in self.inter_indices for inner in outer]
        self.car_indices[inter_list] = False
        self.car_waits = self.car_indices*1
        return self.observation(), seed


    def render_cli(self):
        print('')
        # Render layout as a list of characters
        layout_str = []
        for k in range(len(self.layout)):
            if self.layout[k] == 0:
                layout_str.append('█')
            else:
                layout_str.append(' ')
        # Render spawn and goal blocks
        for k in self.spawn_indices:
            layout_str[k] = '+'
        for k in self.goal_indices:
            layout_str[k] = '@'
        # Render intersection blocks
        for i in range(len(self.inter_indices)):
            str_i = '║' if self.lights[i] == 1 else '═'
            for k in self.inter_indices[i]:
                #layout_str[k] = str(i)
                layout_str[k] = str_i
        # Render cars
        for k in range(len(layout_str)):
            if self.car_indices[k]:
                layout_str[k] = self.car_dict[self.layout_steps[k]]
        # Join list strings, one per line
        for i in range(self.ly):
            print(''.join(layout_str[i::self.ly]))
        print('')
        print(self.get_reward())
    
    
    def render(self):
        self.render_cli()
        




# te = TrafficEnv()
# te.render()

# te.step([0])
# te.render()