import numpy as np

class RewardCalculator:
    def __init__(self, solution_data):
        """
        Initialize the reward calculator with PDE solution data.
        solution_data: A NumPy array of shape (nx, ny, timesteps) representing the PDE solution.
        """
        self.solution_data = solution_data
        self.nx, self.ny, self.timesteps = solution_data.shape
        self.cov_matrix = None
        self.eigenvalues = None
        self.eigenvectors = None
        self.selected_modes = None
        self.reward = None

    def compute_covariance_matrix(self):
        """
        Compute the spatial covariance matrix from the PDE solution.
        """
        # Reshape solution data: (nx*ny, timesteps)
        reshaped_data = self.solution_data.reshape(self.nx * self.ny, self.timesteps)
        # Compute mean over time for each spatial point
        mean_values = np.mean(reshaped_data, axis=1, keepdims=True)
        # Subtract mean to center the data
        centered_data = reshaped_data - mean_values
        # Compute covariance matrix (spatial correlations)
        self.cov_matrix = np.dot(centered_data, centered_data.T) / (self.timesteps - 1)
        return self.cov_matrix

    def solve_eigenvalue_problem(self):
        """
        Solve the eigenvalue problem for the covariance matrix.
        Returns the eigenvalues and eigenvectors, sorted in descending order.
        """
        #print("lol")
        if self.cov_matrix is None:
            raise ValueError("Covariance matrix is not set. Call compute_covariance_matrix() first.")
        
        # Ensure the covariance matrix is symmetric
        self.cov_matrix = (self.cov_matrix + self.cov_matrix.T) / 2
        
        # Compute eigenvalues and eigenvectors
        try:
            eigenvalues, eigenvectors = np.linalg.eigh(self.cov_matrix)
        except np.linalg.LinAlgError as e:
            raise ValueError(f"Eigenvalue decomposition failed: {e}")
        
        # Sort eigenvalues and corresponding eigenvectors in descending order
        sorted_indices = np.argsort(eigenvalues)[::-1]
        self.eigenvalues = eigenvalues[sorted_indices]
        self.eigenvectors = eigenvectors[:, sorted_indices]
        
        return self.eigenvalues, self.eigenvectors

    def select_KLD_modes(self, num_modes):

        if self.eigenvectors is None or self.eigenvalues is None:
            raise ValueError("Eigenvalues must be computed first using solve_eigenvalue_problem().")
        
        # Select the top `num_modes` eigenvectors
        self.selected_modes = self.eigenvectors[:, :num_modes]
        return self.selected_modes
    
    def compute_reward_function(self, sensor_indices):
        
        if self.selected_modes is None:
            raise ValueError("KLD modes must be selected first using select_KLD_modes().")
        
        # Extract the rows of the selected KLD modes corresponding to sensor locations
        selected_sensors_matrix = self.selected_modes[sensor_indices, :]
        
        # Compute the information gain measure (determinant of reduced observability Gramian)
        reward = np.linalg.det(selected_sensors_matrix @ selected_sensors_matrix.T)
        
        return reward