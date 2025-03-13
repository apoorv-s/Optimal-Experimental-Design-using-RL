import numpy as np
from sklearn.decomposition import PCA

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
        print("change 3 \n")
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
        """
        Pm_phi = self.selected_modes[sensor_indices, :]
        reward = np.linalg.det(Pm_phi.T @ Pm_phi)
        return reward
        """
    
    # New PCA-based methods
    def compute_kld_with_pca(self, energy_threshold=0.99):

        # Reshape data for PCA: (timesteps, nx*ny)
        X = self.solution_data.reshape(self.nx * self.ny, self.timesteps).T
        # Fit PCA
        pca = PCA()
        pca.fit(X)
    
        explained_variance_ratio = pca.explained_variance_ratio_
        cumulative_variance_ratio = np.cumsum(explained_variance_ratio)
        n_components = np.searchsorted(cumulative_variance_ratio, energy_threshold) + 1
        
        # Get eigenvalues and eigenvectors (KL modes)
        eigenvalues = pca.explained_variance_[:n_components]
        
        # PCA components have shape (n_components, nx*ny), but we want (nx*ny, n_components)
        modes = pca.components_[:n_components].T
        
        # Store the results in the class attributes
        self.eigenvalues_pca = eigenvalues
        self.selected_modes_pca = modes
        
        return modes, eigenvalues, n_components
    
    def compute_reward_function_pca(self, sensor_indices):
        """
        Compute the reward function using PCA-based KLD modes.
        """
        if not hasattr(self, 'selected_modes_pca'):
            raise ValueError("PCA modes not computed. Call compute_kld_with_pca() first.")
        


        # Extract the rows of the selected PCA modes corresponding to sensor locations
        selected_sensors_matrix = self.selected_modes_pca[sensor_indices, :]
        
        # Compute the information gain measure (determinant of reduced observability Gramian)
        reward = np.linalg.det(selected_sensors_matrix @ selected_sensors_matrix.T)        
        """""
        Pm_phi = self.selected_modes_pca[sensor_indices, :]
        observability_gramian = Pm_phi.T @ Pm_phi
        reward = np.linalg.det(observability_gramian)
        """""
        return reward