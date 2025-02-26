import gymnasium as gym
import numpy as np
from gymnasium import spaces
from sklearn.decomposition import PCA
from pde.AdvectionEquation import  *

class SensorOptimalPlacement(gym.Env):
    metadata = {"render_modes": [None]} #todo: enable render later
    def __init__(self, width, length, n_sensor,seed):
        super().__init__()
        self.width = width # Must be consistent with pde_config
        self.length = length # Must be consistent with pde_config
        self.grid_size = width * length
        self.seed = seed
        self.n_sensor = n_sensor
        #define observation space
        #observation space is a grid with 1 and 0 depicting positions of sensors
        self.observation_space = spaces.MultiBinary((self.width, self.length), seed = self.seed)
        # define action space
        # fof action, at each step, we choose a sensor and move to a new available positions => num_actions  = n_sensor * (width * length - n_sensor)
        # an action is a discrete integer
        self.action_space = spaces.Discrete(self.n_sensor * (self.grid_size - self.n_sensor))

        #initiate the pde dynamics
        self.pde_config = Adv2dModelConfig()
        self.pde_solver= Advection2D(config=self.pde_config)

        #internal tracking
        self.state = None  # Will be a (width x length) array of 0/1
        self.sensor_positions = None  # List of (row, col) for each sensor
        self.pde_field = None #PDE solution field, of shape (n_step, width, length) # n_step defined in pde_config


    def reset(self):
        """initiate new episode by randomly placing n_sensor sensors on the grid"""
        # Create an empty grid
        self.state = np.zeros((self.width, self.length), dtype=int)
        # Choose n_sensor unique cells to place sensors
        flat_indices = np.random.choice(self.grid_size, self.n_sensor, replace=False)

        # Convert flat indices (0..width*length-1) to 2D (row, col)
        self.sensor_positions = [divmod(idx, self.length) for idx in flat_indices]

        # Mark these positions in the state
        for (r, c) in self.sensor_positions:
            self.state[r, c] = 1

        #set initial condition for pde solver
        pde_initial_cond =  self.pde_solver.initial_condition()
        self.pde_field = self.pde_solver.step(pde_initial_cond)

        return self.state

    def step(self,action):
        #update the grid with new placement
        #decode the action integer into what sensor to move, and to what empty lot
        total_empty_spots = self.grid_size - self.n_sensor
        sensor_index = action // total_empty_spots
        empty_spot_index = action % total_empty_spots
        #To understand this, an example is width = 5, length = 5, n_sensor = 5 => total_empty_spots = 20
        #a value of action from 0 to 19 means moving the first sensor (action // 20) to a new empty position 0-19 (action % 20)

        #get all empty spots idx
        empty_positions = np.argwhere(self.state == 0)
        #pick new position
        new_pos = empty_positions[empty_spot_index]
        #selected sensor position
        old_pos = self.sensor_positions[sensor_index]
        #update the grid state and sensor positions
        self.state[old_pos[0], old_pos[1]] = 0
        self.state[new_pos[0], new_pos[1]] = 1
        self.sensor_positions[sensor_index] = tuple(new_pos)

        #Note that we already update pde_solver in the reset step, so we use self.pde_field to evaluate reward before stepping again
        #compute rewards
        reward = self._compute_reward(self.pde_field, self.state, self.sensor_positions)

        #step pde_solver for next step, use the last frame of current pde_field as current state
        current_state = self.pde_field[-1]
        self.pde_field = self.pde_solver.step(current_state)

        #define terminal condition (not sure?)
        done = False
        info = {}
        return self.state, reward, done, info

    def _compute_reward(self, pde_field, grid_state, sensor_positions):
        #run KL decomposition, then calculate criteria score
        pass

    def _karhunen_loeve_pca(pde_output, n_components=None):
        """For discrete dataset/dynamics, KL decomposition is equivalent to PCA"""
        n_step, x_dim, y_dim = pde_output.shape
        reshaped_data = pde_output.reshape(n_step, -1)  # Flatten spatial dimensions

        pca = PCA(n_components=n_components)
        principal_components = pca.fit_transform(reshaped_data)

        return pca.mean_, pca.explained_variance_, pca.components_

    def render(self):
        pass
