import numpy as np

class Adv2dModelConfig():
    def __init__(self):
        self.nx = 50
        self.ny = 50
        self.num_eqn = 1
        
        self.x_velocity = 0.5
        self.y_velocity = 1.0
        
        self.x_domain = [0.0, 1.0]
        self.y_domain = [0.0, 1.0]
        
        self.t_final = 2.0
        self.n_steps = 20
        
        # Pyclaw specific configs
        self.dimensional_split = 1
        self.transverse_waves = 0