import numpy as np
import gymnasium as gym
from gymnasium import spaces
from reward.reward import RewardCalculator


class OEDGymConfig():
    def __init__(self):
        self.n_sensor = 5
        self.n_max = 10
        self.max_horizon = 500
        self.use_pca = True
        self.with_replacement_oed = True


class CustomMultiBinary(spaces.MultiBinary):
    def __init__(self, shape, n_sensor, seed=None):
        super().__init__(shape, seed=seed)
        self.n_sensor = n_sensor

    def sample(self):
        flat_grid = np.zeros(np.prod(self.shape), dtype=int)
        ones_indices = self.np_random.choice(np.prod(self.shape), self.n_sensor, replace=False)
        flat_grid[ones_indices] = 1
        return flat_grid.reshape(self.shape)


class OED(gym.Env):
    metadata = {"render_modes": [None]}
    def __init__(self, seed, pde_system, gym_config: OEDGymConfig):
                #  n_sensors, max_horizon, n_max,
                #  use_pca = True, with_replacement_oed = True):
        # with replacement OED means that multiple sensor can be placed at the same location
        super().__init__()
        self.seed = seed
        np.random.seed(seed)
        
        self.pde_system = pde_system
        self.width = pde_system.nx
        self.length = pde_system.ny
        self.grid_size = self.width * self.length
        
        self.n_sensors = gym_config.n_sensor
        self.max_horizon = gym_config.max_horizon
        self.n_max = gym_config.n_max
        self.use_pca = gym_config.use_pca
        self.with_replacement_oed = gym_config.with_replacement_oed
        
        # TODO: Does using the same seed for the observation space and action space make sense? Is there a reason?
        self.observation_space = CustomMultiBinary((self.width, self.length),
                                              n_sensor=self.n_sensors, seed=self.seed)
        
        # TODO: Should we change it to be n_sensor * grid_size, multiple sensors can converge to the same position! That's okay, might inform the number of sensors needed
        if self.with_replacement_oed:
            self.action_space = spaces.Discrete(self.n_sensors * self.grid_size, seed=self.seed)
        else:
            self.action_space = spaces.Discrete(self.n_sensors * (self.grid_size - self.n_sensors), seed=self.seed)
        
        self.pde_field = self.pde_system.step(self.pde_system.initial_condition())
        self.pde_field = np.transpose(self.pde_field, (1, 2, 0))
        self.reward_calculator = RewardCalculator(self.pde_field)
        
        if self.use_pca:
            self.reward_calculator.compute_kld_with_pca(energy_threshold=0.99)
        else:
            self.reward_calculator.compute_covariance_matrix()
            eigenvalues, _ = self.reward_calculator.solve_eigenvalue_problem()
            cumulative_energy = np.cumsum(eigenvalues) / np.sum(eigenvalues)
            num_modes = np.searchsorted(cumulative_energy, 0.99) + 1
            self.reward_calculator.select_KLD_modes(num_modes)
            
            
        self.state = None  # Sensor position in width x length grid with n_sensors 1s
        self.sensor_positions = None  # List of sensor positions (row_i, col_i)_{i=1:n_sensors}
        self.max_reward = -np.inf
        self.N = 0 # number of steps since reward last improved
        self.optimal_state = None
        self.t = 0
        self.n_episode = 0
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.state = self.observation_space.sample()
        self.sensor_positions = [tuple(pos) for pos in np.argwhere(self.state == 1)] # returns sorted index of 1s
        self.max_reward = -np.inf
        self.optimal_state = None
        self.n_episode += 1
        info = {}        
        return self.state, info
    
    def step(self, action):
        if self.with_replacement_oed:
            sensor_idx = action // self.grid_size
            spot_idx = action % self.grid_size
            
            new_pos = (spot_idx // self.length, spot_idx % self.length)
            old_pos = self.sensor_positions[sensor_idx]
            
        else:
            empty_spots = self.grid_size - self.n_sensors
            sensor_idx = action // empty_spots
            spot_idx = action % empty_spots
            
            empty_pos = np.argwhere(self.state == 0)
            new_pos = empty_pos[spot_idx]
            old_pos = self.sensor_positions[sensor_idx]
            
        self.state[old_pos[0], old_pos[1]] = 0
        self.state[new_pos[0], new_pos[1]] = 1
        self.sensor_positions[sensor_idx] = tuple(new_pos)
        
        reward = self.compute_reward()
        
        if reward > self.max_reward:
            self.max_reward = reward
            self.optimal_state = self.state.copy()
            self.N = 0
        else:
            self.N += 1
            
        self.t += 1
        if (self.N > self.n_max) or (self.t > self.max_horizon):
            done = True
        else:
            done = False

        info = {"optimal_state": self.optimal_state,
                "max_reward": self.max_reward}

        return self.state, reward, done, False, info
        
    def compute_reward(self):
        if self.reward_calculator is None:
            raise ValueError("Reward calculator not initialized.")
        flat_indices = [r * self.length + c for r, c in self.sensor_positions]
        
        if self.use_pca:
            reward = self.reward_calculator.compute_reward_function_pca(flat_indices)
        else:
            reward = self.reward_calculator.compute_reward_function(flat_indices)

        return np.log(reward) if reward > 0 else -float('inf')
    
    def render(self):
        pass

if __name__ == "__main__":
    import IPython
    IPython.embed()


        