import numpy as np
import gymnasium as gym
from gymnasium import spaces
# from reward.reward import RewardCalculator
from sklearn.decomposition import PCA
import copy

class OEDGymConfig():
    def __init__(self):
        self.n_sensor = 5
        self.max_horizon = 500
        # self.use_pca = True
        self.n_components_rewards = 0.99
        self.old_action_space = False


class CustomMultiBinary(spaces.MultiBinary):
    def __init__(self, shape, n_sensor):
        super().__init__(shape)
        self.n_sensor = n_sensor

    def sample(self):
        flat_grid = np.zeros(np.prod(self.shape), dtype=int)
        ones_indices = self.np_random.choice(np.prod(self.shape), self.n_sensor, replace=False)
        flat_grid[ones_indices] = 1
        return flat_grid.reshape(self.shape)


class OED(gym.Env):
    metadata = {"render_modes": [None]}
    def __init__(self, pde_system, gym_config: OEDGymConfig):
                #  n_sensors, max_horizon, n_max,
                #  use_pca = True, with_replacement_oed = True):
        # with replacement OED means that multiple sensor can be placed at the same location
        super().__init__()
        
        self.pde_system = pde_system
        self.nx = pde_system.nx
        self.ny = pde_system.ny
        self.grid_size = self.nx * self.ny
        
        self.n_sensors = gym_config.n_sensor
        self.max_horizon = gym_config.max_horizon
        # self.use_pca = gym_config.use_pca
        self.old_action_space = gym_config.old_action_space

        self.observation_space = CustomMultiBinary((self.nx, self.ny), n_sensor=self.n_sensors)
        
        if self.old_action_space:
            self.action_space = spaces.Discrete(self.n_sensors * (self.grid_size - self.n_sensors))
        else:
            # Action space: for each sensor, 4 possible moves
            self.action_space = spaces.Discrete(4 * self.n_sensors)

        self.pde_field = self.pde_system.step(self.pde_system.initial_condition())
        self.pde_field = np.transpose(self.pde_field, (1, 2, 0))
        
        # Rewards Computation
        self.solution_data = self.pde_field.reshape(-1, self.pde_field.shape[-1]).T
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
        self.max_reward = -np.inf
        self.optimal_state = None
        self.t = 0
        self.n_episode = 0
        
    def reset(self, seed=None, options=None):
        if seed is not None:
            super().reset(seed=seed)
            self.action_space.seed(seed = seed)
            self.observation_space.seed(seed= seed)
        self.state = self.observation_space.sample()
        self.max_reward = -np.inf
        self.optimal_state = None
        self.n_episode += 1
        info = {}
        self.t = 0
        return self.state, info

    def update_state_and_reward(self, current_state, action):
        # move the logic to update state and reward seperately from step() without changing self states to be used by MCTS
        # make a copy to avoid overwritten the current state input
        new_state = copy.deepcopy(current_state)
        new_sensor_positions = [tuple(pos) for pos in np.argwhere(new_state == 1)]
        if self.old_action_space:
            empty_spots = self.grid_size - self.n_sensors
            sensor_idx = action // empty_spots
            spot_idx = action % empty_spots
            empty_pos = np.argwhere(new_state == 0)
            new_pos = empty_pos[spot_idx]
            old_pos = new_sensor_positions[sensor_idx]

            new_state[old_pos[0], old_pos[1]] = 0
            new_state[new_pos[0], new_pos[1]] = 1
            new_sensor_positions[sensor_idx] = tuple(new_pos)

        else:
            sensor_idx = action // 4
            move_idx = action % 4
            # Directions: up, right, down, left
            directions = [(-1, 0), (0, 1), (1, 0), (0, -1)]
            move = directions[move_idx]
            old_pos = new_sensor_positions[sensor_idx]
            new_pos = (old_pos[0] + move[0], old_pos[1] + move[1])

            # Only update if new position is valid: within bounds and unoccupied.
            if (0 <= new_pos[0] < self.nx) and (0 <= new_pos[1] < self.ny):
                if new_state[new_pos[0], new_pos[1]] != 1:
                    new_state[old_pos[0], old_pos[1]] = 0
                    new_state[new_pos[0], new_pos[1]] = 1
                    new_sensor_positions[sensor_idx] = new_pos

        reward = self.compute_reward(new_sensor_positions)
        return new_state, reward

    def step(self, action):
        new_state, reward = self.update_state_and_reward(self.state, action)
        self.state = new_state
        if reward > self.max_reward:
            self.max_reward = reward
            self.optimal_state = self.state.copy()
            
        self.t += 1
        done = (self.t > self.max_horizon)

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
    
    def compute_reward(self, sensor_positions):
        # print(self.sensor_positions)
        sensor_idx = [r * self.ny + c for r, c in sensor_positions]
        Q_m = self.modes[sensor_idx, :]*10
        # T_m = Q_m.T @ Q_m # Symmetric matrix
        reward = np.linalg.det(Q_m.T @ Q_m)
        return reward
        # return np.log(reward) if reward > 0 else -float('inf')
    
    def render(self):
        pass

if __name__ == "__main__":
    import IPython
    IPython.embed()


        