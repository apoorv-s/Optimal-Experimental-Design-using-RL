import gymnasium as gym
import numpy as np
from gymnasium import spaces
from sklearn.decomposition import PCA
from pde.AdvectionEquation import  *
from reward.reward import RewardCalculator

class SensorOptimalPlacement(gym.Env):
    metadata = {"render_modes": [None]} #todo: enable render later
    def __init__(self, width, length, n_sensor, max_horizon, N_max, seed, use_pca = True):
        super().__init__()
        self.width = width # Must be consistent with pde_config
        self.length = length # Must be consistent with pde_config
        self.grid_size = width * length
        self.seed = seed
        np.random.seed(seed)
        self.n_sensor = n_sensor
        self.max_horizon = max_horizon #max horizon by which an epsisode will be terminated
        self.N_max = N_max #max iteration of nonincreasing reward that an episode will be terminated
        #define observation space
        #observation space is a grid with 1 and 0 depicting positions of sensors
        self.observation_space = spaces.MultiBinary((self.width, self.length), seed = self.seed)
        # define action space
        # fof action, at each step, we choose a sensor and move to a new available positions => num_actions  = n_sensor * (width * length - n_sensor)
        # an action is a discrete integer
        self.action_space = spaces.Discrete(self.n_sensor * (self.grid_size - self.n_sensor), seed= self.seed)

        #initiate the pde dynamics
        pde_config = Adv2dModelConfig()
        assert pde_config.nx == width, "x-dimension mismatch"
        assert pde_config.ny == length, "y-dimension mismatch"


        # simulate the dynamics (only once at init of environment)
        pde_solver = Advection2D(config=pde_config)
        pde_initial_cond = pde_solver.initial_condition()
        pde_field = pde_solver.step(pde_initial_cond)
        # Rearranged pde_field to make it work with RewardCalculator
        solution_data = np.transpose(pde_field, (1, 2, 0))
        self.reward_calculator = RewardCalculator(solution_data)
        # Pre-compute with KLD or PCA
        if self.use_pca:
            self.reward_calculator.compute_kld_with_pca(energy_threshold=0.99)
        else:
            # Use original KLD approach
            self.reward_calculator.compute_covariance_matrix()
            eigenvalues, _ = self.reward_calculator.solve_eigenvalue_problem()
            cumulative_energy = np.cumsum(eigenvalues) / np.sum(eigenvalues)
            num_modes = np.searchsorted(cumulative_energy, 0.99) + 1
            self.reward_calculator.select_KLD_modes(num_modes)

        #internal tracking
        self.state = None  # Will be a (width x length) array of 0/1
        self.sensor_positions = None  # List of (row, col) for each sensor
        self.max_reward = -np.inf
        self.N = 0 #number of steps since reward last improved
        self.optimal_state = None
        self.t = 0 #time step in current episode
        self.n_episode = 0

    def reset(self, seed = None, options = None):
        """initiate new episode by randomly placing n_sensor sensors on the grid"""

        super().reset(seed = seed)
        # Create an empty grid
        self.state = np.zeros((self.width, self.length), dtype=int)
        # Choose n_sensor unique cells to place sensors
        flat_indices = np.random.choice(self.grid_size, self.n_sensor, replace=False)

        # Convert flat indices (0..width*length-1) to 2D (row, col)
        self.sensor_positions = [divmod(idx, self.length) for idx in flat_indices]

        # Mark these positions in the state
        for (r, c) in self.sensor_positions:
            self.state[r, c] = 1


        #reset tracking state
        self.N = 0
        self.t = 0
        self.max_reward = -np.inf
        self.optimal_state = None
        self.n_episode += 1

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
        reward = self._compute_reward()

        #update max reward
        if reward > self.max_reward:
            self.max_reward = reward
            self.optimal_state = self.state.copy()
            self.N = 0
        else:
            self.N += 1

        #define terminal condition
        self.t += 1
        if (self.N > self.N_max) or (self.t > self.max_horizon):
            done = True
        else:
            done = False

        info = {"optimal_state": self.optimal_state,
                "max_reward": self.max_reward}

        return self.state, reward, done, False, info

    def _compute_reward(self):
        """Compute reward for current sensor configuration using pre-computed KLD/PCA modes"""
        if self.reward_calculator is None:
            raise ValueError("Reward calculator not initialized. Call reset() first.")

        flat_indices = [r * self.length + c for r, c in self.sensor_positions]

        # The final reward using the precomputed modes (PCA or KLD)
        if self.use_pca:
            reward = self.reward_calculator.compute_reward_function_pca(flat_indices)
        else:
            reward = self.reward_calculator.compute_reward_function(flat_indices)

        return np.log(reward) if reward > 0 else -float('inf')

    def render(self):
        pass
