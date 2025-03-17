from src.OED import OED, OEDGymConfig
from src.DQN import DQN_OED
from src.MCTS import *
from pde.AdvectionEquation import *
from pde.Burgers2D import *
from pde.AdvectionDiffusionReaction import *

if __name__ == "__main__":
    seed = 0
    adv_config = Adv2dModelConfig()
    adv_eq = Advection2D(adv_config)
    gym_config = OEDGymConfig()
    gym_config.n_max = 100
    gym_config.max_horizon = 1000
    mcts_config = MCTSConfig()
    mcts = MCTS(seed, adv_eq,gym_config, mcts_config)
    mcts.train()