import numpy as np
import fipy as fp

class ADRConfig():
    def __init__(self):
        self.nx = 50
        self.ny = 50
        
        self.x_domain = [0.0, 1.0]
        self.y_domain = [0.0, 1.0]
        
        self.t_final = 3.0
        self.delta_t = 0.001
        self.n_steps = int(self.t_final / self.delta_t)
        self.t_steps = np.arange(0, self.t_final + self.delta_t, self.delta_t)
        
        self.diff_coeff = 0.02
        self.adv_coeff = (0.1, 0.1)
        self.react_coeff = -0.01
        
        self.x0 = 0.25
        self.y0 = 0.25
        
        
    
class ADR():
    def __init__(self, config:ADRConfig):
        self.nx = config.nx
        self.ny = config.ny
        self.x_domain = config.x_domain
        self.y_domain = config.y_domain
        
        self.len_x = self.x_domain[1] - self.x_domain[0]
        self.len_y = self.y_domain[1] - self.y_domain[0]
        
        self.dx = self.len_x / self.nx
        self.dy = self.len_y / self.ny
        
        self.n_steps = config.n_steps
        self.delta_t = config.delta_t
        self.t_steps = config.t_steps
        
        self.x0 = config.x0
        self.y0 = config.y0
        
        self.mesh = fp.Grid2D(dx=self.dx, dy=self.dy, nx=self.nx, ny=self.ny)
        
        self.X, self.Y = self.mesh.cellCenters
        self.X_grid = self.X.reshape((self.nx, self.ny))
        self.Y_grid = self.Y.reshape((self.nx, self.ny))
        
        self.diff_coeff = config.diff_coeff
        self.adv_coeff = config.adv_coeff
        self.react_coeff = config.react_coeff
        
    
    def step(self, current_state):
        u = fp.CellVariable(name="concentration", mesh=self.mesh, value=0.0)
        
        eq = fp.TransientTerm() + fp.ConvectionTerm(self.adv_coeff) - fp.DiffusionTerm(self.diff_coeff) - self.react_coeff * u
        
        X, Y = self.mesh.cellCenters
        u.setValue(current_state)
        
        u.faceGrad.constrain(0, where=self.mesh.exteriorFaces)
        
        results = np.zeros((self.n_steps + 1, self.nx, self.ny))
        results[0] = u.value.reshape((self.nx, self.ny)).copy()
        for step in range(self.n_steps):
            eq.solve(var=u, dt=self.delta_t)
            results[step + 1] = u.value.reshape((self.nx, self.ny)).copy()
            
        return results
    
    def initial_condition(self):
        # Unlike the Pyclaw examples, this returns a vector instead of a 2D array
        return 5*np.exp(-((self.X - self.x0) ** 2 + (self.Y - self.y0) ** 2) / 0.02)