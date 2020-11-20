from gym.utils import seeding
from gym import spaces
import numpy as np


class TrafficEnv():
    def __init__(self, horiz_lanes=('e','w'), vert_lanes=('n','s','sn'), horiz_sizes=(3,3,4,2), vert_sizes=(3,3,3), 
                    car_speed=2, max_wait=100, max_wait_penalty=1000000, max_steps=100):
        self.horiz_lanes = horiz_lanes
        self.vert_lanes = vert_lanes
        self.horiz_sizes = horiz_sizes
        self.vert_sizes = vert_sizes
        self.car_speed = car_speed
        self.max_wait = max_wait
        self.max_wait_penalty = max_wait_penalty
        self.max_steps = max_steps
        
        self.verify_inputs()
        self.get_layout()
        
        self.has_inf_speed = self.car_speed == 0
        if self.has_inf_speed:
            self.get_waitlines()
        
        self.action_space = spaces.MultiDiscrete(
            [2 for _ in range(len(self.horiz_lanes) * len(self.vert_lanes))])
        if self.has_inf_speed:
            self.observation_space = spaces.MultiDiscrete(
                [x + 1 for x in self.waitline_sizes]
                + [max_wait + 1 for _ in range(len(self.horiz_lanes) + len(self.vert_lanes))])
        else:
            self.observation_space = spaces.MultiDiscrete(
                [2 for _ in range(self.valid_car_indices.size)]
                + [max_wait + 1 for _ in range(len(self.horiz_lanes) + len(self.vert_lanes))])
        # Make all lights green for vertical lanes ('1'), it does not matter
        self.lights = [1 for _ in range(len(self.horiz_lanes) * len(self.vert_lanes))]
        
        # self.make_spawn_blocks(self.start_indices, [0.5 for _ in range(len(self.start_indices))])
        # if self.has_inf_speed:
            # self.reset(self.waitline_sizes)
        # else:
            # self.reset([True for _ in range(len(self.valid_car_indices))])
        
    
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
            raise TypeError('Argument \'car_speed\' must be a non-negative int')
        elif self.car_speed < 0:
            raise ValueError('Argument \'car_speed\' is negative')
        # Verify 'max_wait'
        if type(self.max_wait) != int:
            raise TypeError('Argument \'max_wait\' must be a positive int')
        elif self.max_wait <= 0:
            raise ValueError('Argument \'max_wait\' is not positive')
        # Verify 'max_wait_penalty'
        if type(self.max_wait_penalty) != int:
            raise TypeError('Argument \'max_wait_penalty\' must be a positive int')
        elif self.max_wait_penalty <= 0:
            raise ValueError('Argument \'max_wait_penalty\' is not positive')
        # Verify 'max_steps'
        if type(self.max_steps) != int:
            raise TypeError('Argument \'max_steps\' must be a positive int')
        elif self.max_steps <= 0:
            raise ValueError('Argument \'max_steps\' is not positive')

    
    def reward(self):
        return self.last_action_dist if \
            self.max_cum_wait < self.max_wait else -self.max_wait_penalty

    
    def observation(self):
        if self.has_inf_speed:
            # Backtrack each waitline counting the number of cars waiting
            observation = []
            for i in range(len(self.waitline_indices)):
                ind = self.waitline_indices[i]
                ind_step = self.layout_steps[ind]
                n_cars = 0
                for k in range(self.waitline_sizes[i]):
                    if self.car_indices[ind-k*ind_step]:
                        n_cars += 1
                    else:
                        break
                observation.append(n_cars)
        else:
            observation = self.car_indices[self.valid_car_indices].tolist()
        observation += self.horiz_cum_wait.tolist() + self.vert_cum_wait.tolist()
        return observation
    
        
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
            return False, ind_next
        elif self.car_indices[ind_next]:
            # Next block is occupied
            return False, ind
        else:
            # Car can move to next block
            self.car_indices[ind] = False
            self.car_indices[ind_next] = True
            return True, ind_next
    
    
    def enter_car(self):
        # Generate new cars
        new_car_indices = []
        for i in range(len(self.spawn_indices)):
            s = self.spawn_indices[i]
            ind_step = self.layout_steps[s]
            p = self.arrival_rate[i]
            kmax = 1 if self.has_inf_speed else self.car_speed
            for k in range(kmax):
                ind_k = s+k*ind_step
                if self.car_indices[ind_k]:
                    break
                else:
                    self.car_indices[ind_k] = self.np_random.binomial(1,p) == 1
                    if self.car_indices[ind_k]:
                        new_car_indices.append(ind_k)
        return new_car_indices
    
    
    def step(self, action):
        # We save some overhead by not verifying input action
        self.lights = action
        self.last_action_dist = 0
        # Make stack of cars by going through lanes in reverse
        car_stack = []
        for spawn_k in self.start_indices:
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
            k = 0
            while True:
                can_move, ind_next = self.move_car(ind)
                if ind_next != ind:
                    k += 1
                if not can_move or (k >= self.car_speed and not self.has_inf_speed):
                    break
                ind = ind_next
            self.last_action_dist += k
            if not can_move and ind_next == ind:
                # Car could not move at full speed and did not reach the goal,
                # so the car had to wait
                if self.horiz_indices[ind] == 0:
                    # Vertical street
                    self.vert_cum_wait[self.vert_indices[ind]-1] += 1
                else:
                    # Horizontal street
                    self.horiz_cum_wait[self.horiz_indices[ind]-1] += 1
        # Substract minimum cumulative waiting from the rest
        min_cum_wait = min((min(self.horiz_cum_wait), min(self.vert_cum_wait)))
        self.horiz_cum_wait -= min_cum_wait
        self.vert_cum_wait -= min_cum_wait
        # Limit cumulative waiting to its max. value
        self.horiz_cum_wait[self.horiz_cum_wait > self.max_wait] = self.max_wait
        self.vert_cum_wait[self.vert_cum_wait > self.max_wait] = self.max_wait
        # Compute maximum cumulative waiting
        self.max_cum_wait = max((max(self.horiz_cum_wait), max(self.vert_cum_wait)))
        # Let new cars join
        new_car_stack = self.enter_car()
        if self.has_inf_speed:
            # Move new cars until stack is empty (just if speed is infinite)
            # Reward and cumulative waitings are not affected this time
            len_stack = len(new_car_stack)
            while len_stack > 0:
                ind = new_car_stack.pop()
                len_stack -= 1
                while True:
                    can_move, ind_next = self.move_car(ind)
                    if not can_move:
                        break
                    ind = ind_next
        # Advance one step
        self.current_step += 1
        # Compute output variables
        observation = self.observation()
        reward = self.reward()
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
        # Create matrix with horizontal street indices (1-indexed)
        self.horiz_indices = np.zeros((self.ly, self.lx), dtype=np.int32)
        # Create matrix with vertical street indices (1-indexed)
        self.vert_indices = np.zeros((self.ly, self.lx), dtype=np.int32)
        # List of start blocks, marking the start of a lane
        self.start_indices = []
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
                    self.start_indices.append(ind)
                    self.goal_indices.append(ind + self.ly * (self.lx - 1))
                else:
                    self.layout[ind,:] = ind + self.ly * (np.arange(0, self.lx) - 1)
                    self.layout[ind,0] = 0
                    self.layout_steps[ind,:] = 3
                    self.start_indices.append(ind + self.ly * (self.lx - 1))
                    self.goal_indices.append(ind)
                self.horiz_indices[ind,:] = k + 1
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
                    self.start_indices.append(self.ly * (ind + 1) - 1)
                    self.goal_indices.append(self.ly * ind)
                else:
                    self.layout[:,ind] = self.ly * ind + (np.arange(0, self.ly) + 1)
                    self.layout[-1,ind] = 0
                    self.layout_steps[:,ind] = 4
                    self.start_indices.append(self.ly * ind)
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
                self.vert_indices[:,ind] = k + 1
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
        # Delete step size at intersection blocks
        inter_list = [inner for outer in self.inter_indices for inner in outer]
        self.layout_steps[inter_list] = 0
        # Get logic index vector of valid car positions
        self.valid_car_indices = np.logical_and(
            self.layout_steps != 0, self.layout != 0).nonzero()[0]
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
        # Convert street index matrices to vectors
        self.horiz_indices = np.ravel(self.horiz_indices, order='F')
        self.vert_indices = np.ravel(self.vert_indices, order='F')
    
    
    def get_waitlines(self):
        self.waitline_indices = []
        self.waitline_sizes = []
        # Traverse each lane from its start block
        for start in self.start_indices:
            ind = start
            # Get car direction
            ind_step = self.layout_steps[ind]
            # Start traversing lane
            size_k = 1
            while True:
                # Get next block
                ind_next = ind + ind_step
                block_next = self.layout[ind]
                if block_next < 0:
                    # Car is entering an intersection, lane segment finished
                    while self.layout[ind_next] == block_next:
                        ind_next += ind_step
                    self.waitline_indices.append(ind)
                    self.waitline_sizes.append(size_k)
                    size_k = 0
                if self.layout[ind_next] == 0:
                    # Car reached a goal block, this segment is not included
                    break
                # Advance to next block
                size_k += 1
                ind = ind_next
    
    
    def make_spawn_blocks(self, spawn_indices, arrival_rate):
        # This method must be called before using the object!
        # Verify input 'spawn_indices'
        if type(spawn_indices) != tuple and type(spawn_indices) != list \
            and type(spawn_indices) != np.ndarray:
            raise TypeError('Argument \'spawn_indices\' must be a tuple or list or numpy.ndarray')
        else:
            for k in range(len(spawn_indices)):
                if type(spawn_indices[k]) != int:
                    raise TypeError('Entry %d of argument \'spawn_indices\' is not an int' % k)
                elif spawn_indices[k] < 0:
                    raise ValueError('Entry %d of argument \'spawn_indices\' is negative' % k)
        self.spawn_indices = np.array(spawn_indices)
        # Verify input 'arrival_rate'
        if type(arrival_rate) != tuple and type(arrival_rate) != list \
            and type(arrival_rate) != np.ndarray:
            raise TypeError('Argument \'arrival_rate\' must be a tuple or list or numpy.ndarray')
        elif len(arrival_rate) != len(spawn_indices):
            raise ValueError(
                'Argument \'arrival_rate\' must have the length of argument \'spawn_indices\'')
        else:
            for k in range(len(arrival_rate)):
                if arrival_rate[k] < 0.0 or arrival_rate[k] > 1.0:
                    raise ValueError(
                        'Entry %d of argument \'arrival_rate\' is not in the range [0,1]' % k)
        self.arrival_rate = np.array(arrival_rate)
        if len(self.spawn_indices) != 0:
            # Remove indices of invalid blocks
            ind_valid = self.layout[self.spawn_indices] > 0
            self.spawn_indices = self.spawn_indices[ind_valid]
            self.arrival_rate = self.arrival_rate[ind_valid]
            # Remove repeated indices
            self.spawn_indices, ind_unique = np.unique(self.spawn_indices, return_index=True)
            self.arrival_rate = self.arrival_rate[ind_unique]
    
    
    def reset(self, cars, seed=None):
        # Verify input
        if type(cars) != tuple and type(cars) != list and type(cars) != np.ndarray:
            raise TypeError('Argument \'cars\' must be a tuple or list or numpy.ndarray')
        elif len(cars) <= 0:
            raise ValueError('Argument \'cars\' is empty')
        # elif self.has_inf_speed:
            # for k in range(len(cars)):
                # if type(cars[k]) != int:
                    # raise TypeError('Entry %d of argument \'cars\' is not an int' % k)
                # elif cars[k] < 0:
                    # raise ValueError('Entry %d of argument \'cars\' is negative' % k)
        # else:
            # for k in range(len(cars)):
                # if type(cars[k]) != bool:
                    # raise TypeError('Entry %d of argument \'cars\' is not a bool' % k)
        # Reset cars
        self.car_indices = np.full(self.layout.shape, False)
        if self.has_inf_speed:
            # Backtrack each waitline adding the specified number of cars
            for i in range(len(self.waitline_indices)):
                ind = self.waitline_indices[i]
                ind_step = self.layout_steps[ind]
                for k in range(min((cars[i], self.waitline_sizes[i]))):
                    self.car_indices[ind-k*ind_step] = True
        else:
            self.car_indices[self.valid_car_indices] = cars[:self.valid_car_indices.size]
        # Reset waiting counters
        self.horiz_cum_wait = np.array([0 for _ in range(len(self.horiz_lanes))])
        self.vert_cum_wait = np.array([0 for _ in range(len(self.vert_lanes))])
        self.last_action_dist = 0
        # Reset step counters
        self.current_step = 0
        # Reseed random generator
        self.np_random, seed = seeding.np_random(seed)
        #return self.observation(), seed
        return self.observation()


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
        # # Render waitline first blocks
        # for k in self.waitline_indices:
            # layout_str[k] = '*'
        # Join list strings, one per line
        for i in range(self.ly):
            print(''.join(layout_str[i::self.ly]))
        print('')
    
    
    def render(self):
        self.render_cli()


    def muzero_reset(self):
        if self.has_inf_speed:
            val = self.reset(self.waitline_sizes)
        else:
            val = self.reset([True for _ in range(len(self.valid_car_indices))])
        return val


    def to_play(self):
        return 1


    def legal_actions(self):
        return list(range(2**self.action_space.shape[0]))


    def step_numerical_action(self, action):
        num_lights = len(self.action_space.sample())
        action = np.array([int(x) for x in bin(action)[2:]])
        action = np.concatenate((np.zeros(num_lights),action))
        action = action[-num_lights:]
        return step(self.action)