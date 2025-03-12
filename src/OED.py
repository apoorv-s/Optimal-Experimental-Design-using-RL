import numpy as np
import gymnasium as gym
from gymnasium import spaces
# from reward.reward import RewardCalculator
from sklearn.decomposition import PCA


class OEDGymConfig():
    def __init__(self):
        self.n_sensor = 5
        self.n_max = 10
        self.max_horizon = 500
        # self.use_pca = True
        self.with_replacement_oed = True
        self.n_components_rewards = 10


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
        self.nx = pde_system.nx
        self.ny = pde_system.ny
        self.grid_size = self.nx * self.ny
        
        self.n_sensors = gym_config.n_sensor
        self.max_horizon = gym_config.max_horizon
        self.n_max = gym_config.n_max
        # self.use_pca = gym_config.use_pca
        self.with_replacement_oed = gym_config.with_replacement_oed

        # TODO: Does using the same seed for the observation space and action space make sense? Is there a reason?
        self.observation_space = CustomMultiBinary((self.nx, self.ny),
                                              n_sensor=self.n_sensors, seed=self.seed)

        if self.with_replacement_oed:
            # Action space: for each sensor, 5 possible moves
            self.action_space = spaces.Discrete(5 * self.n_sensors, seed=self.seed)
        else:
            # Action space: for each sensor, 4 possible moves
            self.action_space = spaces.Discrete(4 * self.n_sensors, seed=self.seed)

        self.pde_field = self.pde_system.step(self.pde_system.initial_condition())
        self.pde_field = np.transpose(self.pde_field, (1, 2, 0))
        
        # Rewards Computation
        self.solution_data = self.pde_field.reshape(-1, self.pde_field.shape[-1])
        self.pca = PCA(n_components=gym_config.n_components_rewards)
        self.pca.fit(self.solution_data)
        self.modes = self.pca.components_.T
        
        # self.reward_calculator = RewardCalculator(self.pde_field)
        
        # if self.use_pca:
        #     self.reward_calculator.compute_kld_with_pca(energy_threshold=0.99)
        # else:
        #     self.reward_calculator.compute_covariance_matrix()
        #     eigenvalues, _ = self.reward_calculator.solve_eigenvalue_problem()
        #     cumulative_energy = np.cumsum(eigenvalues) / np.sum(eigenvalues)
        #     num_modes = np.searchsorted(cumulative_energy, 0.99) + 1
        #     self.reward_calculator.select_KLD_modes(num_modes)
            
            
        self.state = None  # Sensor position in nx x ny grid with n_sensors 1s
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
        self.t = 0
        self.N = 0
        return self.state, info
    
    def step(self, action):
        if self.with_replacement_oed:
            # Decode which sensor and which direction
            sensor_idx = action // 5
            move_idx = action % 5

            # Directions: up, right, down, left
            # up    = (row - 1, col)
            # right = (row, col + 1)
            # down  = (row + 1, col)
            # left  = (row, col - 1)
            # stand-by = (row + 0, col + 0)
            directions = [(-1, 0), (0, 1), (1, 0), (0, -1), (0,0)]

        else:
            # Decode which sensor and which direction
            sensor_idx = action // 4
            move_idx = action % 4

            # Directions: up, right, down, left
            # up    = (row - 1, col)
            # right = (row, col + 1)
            # down  = (row + 1, col)
            # left  = (row, col - 1)
            directions = [(-1, 0), (0, 1), (1, 0), (0, -1)]

        move = directions[move_idx]
        old_pos = self.sensor_positions[sensor_idx]
        new_pos = (old_pos[0] + move[0], old_pos[1] + move[1])

        # Check boundary, ignore the action if out of grid bound
        if (0 <= new_pos[0] < self.nx) and (0 <= new_pos[1] < self.ny):
            #if the new cell is already occupied, do nothing, or penalize?
            if self.state[new_pos[0], new_pos[1]] == 1:
                pass
            else:
                # Valid, unoccupied move -> update
                self.state[old_pos[0], old_pos[1]] = 0
                self.state[new_pos[0], new_pos[1]] = 1
                self.sensor_positions[sensor_idx] = new_pos
        else:
            # Out of bounds – do nothing, or penalize?
            pass

        reward = self.compute_reward()
        
        if reward > self.max_reward:
            self.max_reward = reward
            self.optimal_state = self.state.copy()
            self.N = 0
        else:
            self.N += 1
            
        self.t += 1
        done = (self.N > self.n_max) or (self.t > self.max_horizon)

        info = {"optimal_state": self.optimal_state,
                "max_reward": self.max_reward}

        return self.state, reward, done, False, info
        
    # def compute_reward(self):
    #     if self.reward_calculator is None:
    #         raise ValueError("Reward calculator not initialized.")
    #     flat_indices = [r * self.ny + c for r, c in self.sensor_positions]
        
    #     if self.use_pca:
    #         reward = self.reward_calculator.compute_reward_function_pca(flat_indices)
    #     else:
    #         reward = self.reward_calculator.compute_reward_function(flat_indices)

    #     return np.log(reward) if reward > 0 else -float('inf')
    
    def compute_reward(self):
        print(self.sensor_positions)
        sensor_idx = [r * self.ny + c for r, c in self.sensor_positions]
        Q_m = self.modes[sensor_idx, :]
        # T_m = Q_m.T @ Q_m # Symmetric matrix
        reward = np.linalg.det(Q_m.T @ Q_m)
        return np.log(reward) if reward > 0 else -float('inf')
    
    def render(self):
        pass

if __name__ == "__main__":
    import IPython
    IPython.embed()


        