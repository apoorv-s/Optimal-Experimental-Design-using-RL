import numpy as np
from clawpack import riemann, pyclaw


class Burgers2DConfig():
    def __init__(self):
        self.nx = 100
        self.ny = 100

        self.x_domain = [0.0, 1.0]
        self.y_domain = [0.0, 1.0]

        self.t_final = 2.0
        self.n_steps = 21

        # Pyclaw specific configs
        self.dimensional_split = False
        self.transverse_waves = 2


## 2D Inviscid Burger Example
class Burgers2D():
    def __init__(self, config: Burgers2DConfig):
        self.nx = config.nx
        self.ny = config.ny
        self.t_final = config.t_final
        self.n_steps = config.n_steps
        
        self.t_steps = np.linspace(0, self.t_final, self.n_steps)
        
        self.dimensional_split = config.dimensional_split
        self.transverse_waves = config.transverse_waves
        
        self.x_domain = config.x_domain
        self.y_domain = config.y_domain
        
        self.x = pyclaw.Dimension(self.x_domain[0], self.x_domain[1], self.nx, name='x')
        self.y = pyclaw.Dimension(self.y_domain[0], self.y_domain[1], self.ny, name='y')
        
        self.domain = pyclaw.Domain([self.x, self.y])
        
        
    def initial_condition(self, X, Y):
        sigma_x = 0.3
        sigma_y = 0.3
        x0 = 0.5
        y0 = 0.5
        return np.exp(-(((X - x0) ** 2) / (2 * sigma_x ** 2) + ((Y - y0) ** 2) / (2 * sigma_y ** 2))) 
        
    def solve(self):
        ## Solver setup
        solver = pyclaw.ClawSolver2D(riemann.burgers_2D)
        solver.dimensional_split = self.dimensional_split
        solver.transverse_waves = self.transverse_waves
        solver.cfl_max = 1.0
        solver.cfl_desired = 0.9
        solver.limiters = pyclaw.limiters.tvd.MC
        
        solver.bc_lower[0] = pyclaw.BC.periodic
        solver.bc_upper[0] = pyclaw.BC.periodic
        solver.bc_lower[1] = pyclaw.BC.periodic
        solver.bc_upper[1] = pyclaw.BC.periodic
        
        state = pyclaw.State(self.domain, solver.num_eqn)

        # Initial data
        X, Y = state.grid.p_centers
        state.q[0,:,:] = self.initial_condition(X, Y)

        claw = pyclaw.Controller()
        claw.tfinal = self.t_final
        claw.num_output_times = self.n_steps - 1
        claw.solution = pyclaw.Solution(state, self.domain)
        claw.solver = solver
        claw.keep_copy = True
        claw.output_format = None
        claw.verbosity = 0
        claw.run()
        
        state_array = np.zeros((self.n_steps, self.nx, self.ny))
        for i, frame in enumerate(claw.frames):
            state_array[i, :, :] = frame.q[0, :, :]
        return state_array
