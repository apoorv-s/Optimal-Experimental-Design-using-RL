import numpy as np
import matplotlib.pyplot as plt

import random

from deap import base, creator, tools, algorithms
from src.OED import OED, OEDGymConfig

class GAConfig():
    def __init__(self):
        self.population_size = 100
        self.generations = 100

class GA_OED():
    def __init__(self, seed, pde_system, gym_config: OEDGymConfig, ga_config: GAConfig):
        self.env = OED(seed, pde_system, gym_config)
        self.n_sensor = gym_config.n_sensor
        self.grid_size = self.env.grid_size
        self.n1 = self.env.width
        self.n2 = self.env.length
        
        self.population_size = ga_config.population_size
        self.generations = ga_config.generations
        
        self._setup_ga()
        
        
    def _setup_ga(self):
        """Set up the genetic algorithm components."""
        # Create fitness and individual classes
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMax)
        
        self.toolbox = base.Toolbox()
        
        # Define gene initialization function to create binary string
        # Length is n1*n2 where 1s represent sensor positions
        def create_individual():
            # Create a binary string with exactly num_sensors 1s
            individual = [0] * (self.grid_size)
            ones_indices = random.sample(range(self.grid_size), self.n_sensor)
            for idx in ones_indices:
                individual[idx] = 1
            return individual

        
        # Register creation functions
        self.toolbox.register("individual", tools.initIterate, creator.Individual, create_individual)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)
        
        # Register evaluation, crossover, mutation, and selection
        self.toolbox.register("evaluate", self._evaluate)
        self.toolbox.register("mate", self._custom_crossover)
        self.toolbox.register("mutate", self._custom_mutation)
        self.toolbox.register("select", tools.selTournament, tournsize=3)
        
    def _custom_crossover(self, ind1, ind2):
        """
        Custom crossover that maintains exactly num_sensors 1s in each individual.
        """
        # Get positions of 1s in both parents
        ones_ind1 = [i for i, val in enumerate(ind1) if val == 1]
        ones_ind2 = [i for i, val in enumerate(ind2) if val == 1]
        
        # Select crossover point
        crossover_point = random.randint(1, self.n_sensor - 1)
        
        # Create children
        child1 = [0] * (self.n1 * self.n2)
        child2 = [0] * (self.n1 * self.n2)
        
        # First part from parent 1, second part from parent 2
        for idx in ones_ind1[:crossover_point] + ones_ind2[crossover_point:]:
            child1[idx] = 1
            
        # First part from parent 2, second part from parent 1
        for idx in ones_ind2[:crossover_point] + ones_ind1[crossover_point:]:
            child2[idx] = 1
            
        # Fix if we have duplicate positions (not exactly num_sensors 1s)
        child1 = self._fix_individual(child1)
        child2 = self._fix_individual(child2)
        
        # Replace parents with children
        ind1[:] = child1
        ind2[:] = child2
        
        return ind1, ind2
    
    def _fix_individual(self, individual):
        """Ensure individual has exactly num_sensors 1s."""
        ones_count = sum(individual)
        
        if ones_count < self.n_sensor:
            # Add more 1s
            zeros_indices = [i for i, val in enumerate(individual) if val == 0]
            to_add = random.sample(zeros_indices, self.n_sensor - ones_count)
            for idx in to_add:
                individual[idx] = 1
        elif ones_count > self.n_sensor:
            # Remove excess 1s
            ones_indices = [i for i, val in enumerate(individual) if val == 1]
            to_remove = random.sample(ones_indices, ones_count - self.n_sensor)
            for idx in to_remove:
                individual[idx] = 0
                
        return individual
    
    def _custom_mutation(self, individual, indpb=0.1):
        """
        Custom mutation that maintains exactly num_sensors 1s.
        Moves a sensor from one location to another.
        """
        if random.random() < indpb:
            # Get current sensor positions
            ones_indices = [i for i, val in enumerate(individual) if val == 1]
            zeros_indices = [i for i, val in enumerate(individual) if val == 0]
            
            # Select a sensor to move
            to_move = random.choice(ones_indices)
            
            # Select a new position
            new_pos = random.choice(zeros_indices)
            
            # Move the sensor
            individual[to_move] = 0
            individual[new_pos] = 1
            
        return individual,
    
    def _evaluate(self, individual):
        """
        Evaluate the coverage of sensors on the grid.
        Returns a fitness score based on the proportion of the grid covered.
        """
        # Convert flat index to 2D coordinates
        sensor_positions = []
        for i, val in enumerate(individual):
            if val == 1:
                row = i // self.n2
                col = i % self.n2
                sensor_positions.append((row, col))
        
                
        if self.env.reward_calculator is None:
            raise ValueError("Reward calculator not initialized.")
        flat_indices = [r * self.env.length + c for r, c in sensor_positions]
        
        if self.env.use_pca:
            reward = self.env.reward_calculator.compute_reward_function_pca(flat_indices)
        else:
            reward = self.env.reward_calculator.compute_reward_function(flat_indices)

        reward = np.log(reward) if reward > 0 else -float('inf')
        return reward,
    
    def run(self):
        """Run the genetic algorithm to find optimal sensor placement."""
        # Create initial population
        pop = self.toolbox.population(n=self.population_size)
        
        # Hall of Fame to keep track of best individual
        hof = tools.HallOfFame(1)
        
        # Statistics to track
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean)
        stats.register("min", np.min)
        stats.register("max", np.max)
        
        # Run the algorithm
        pop, logbook = algorithms.eaSimple(
            pop, 
            self.toolbox, 
            cxpb=0.7,  # Crossover probability
            mutpb=0.3,  # Mutation probability
            ngen=self.generations, 
            stats=stats, 
            halloffame=hof,
            verbose=True
        )
        
        # Get the best solution
        best_individual = hof[0]
        best_fitness = hof[0].fitness.values[0]
        
        # Convert best individual to sensor positions
        sensor_positions = []
        for i, val in enumerate(best_individual):
            if val == 1:
                row = i // self.n2
                col = i % self.n2
                sensor_positions.append((row, col))
        
        return best_individual, best_fitness, sensor_positions, logbook
    
    def visualize_solution(self, sensor_positions):
        """Visualize the sensor placement and coverage."""
        plt.figure(figsize=(10, 8))
        
        # Create a grid
        grid = np.zeros(self.grid_size)
        
        # Mark sensor positions
        for row, col in sensor_positions:
            grid[row, col] = 2  # Sensors will be marked with 2
        
        # Mark coverage
        for row in range(self.n1):
            for col in range(self.n2):
                # Skip if this cell already has a sensor
                if grid[row, col] == 2:
                    continue
                    
                # Check if this cell is covered by any sensor
                for s_row, s_col in sensor_positions:
                    distance = np.sqrt((row - s_row)**2 + (col - s_col)**2)
                    if distance <= self.coverage_radius:
                        grid[row, col] = 1  # Covered cells will be marked with 1
                        break
        
        # Create custom colormap: 0=white (uncovered), 1=light blue (covered), 2=red (sensor)
        from matplotlib.colors import ListedColormap
        cmap = ListedColormap(['white', 'lightblue', 'red'])
        
        # Plot the grid
        plt.imshow(grid, cmap=cmap, vmin=0, vmax=2)
        
        # Add grid lines
        plt.grid(which='major', axis='both', linestyle='-', color='k', linewidth=1)
        plt.xticks(np.arange(-0.5, self.n2, 1), [])
        plt.yticks(np.arange(-0.5, self.n1, 1), [])
        
        # Add numbers to mark sensor positions
        for i, (row, col) in enumerate(sensor_positions):
            plt.text(col, row, str(i+1), color='white', ha='center', va='center', fontweight='bold')
        
        # Add labels and title
        plt.title(f'Sensor Placement (Coverage Radius: {self.coverage_radius})')
        
        # Add a legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='white', edgecolor='k', label='Uncovered'),
            Patch(facecolor='lightblue', edgecolor='k', label='Covered'),
            Patch(facecolor='red', edgecolor='k', label='Sensor')
        ]
        plt.legend(handles=legend_elements, loc='upper right')
        
        plt.tight_layout()
        plt.show()
        
        