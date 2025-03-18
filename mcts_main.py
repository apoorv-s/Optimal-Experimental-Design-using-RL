from src.OED import OED, OEDGymConfig
from src.DQN import DQN_OED
from src.MCTS import *
from pde.AdvectionEquation import *
from pde.Burgers2D import *
from pde.AdvectionDiffusionReaction import *
import scipy.io

if __name__ == "__main__":
    seed = 0
    adv_config = Adv2dModelConfig()
    adv_eq = Advection2D(adv_config)
    gym_config = OEDGymConfig()
    gym_config.max_horizon = 10
    mcts_config = MCTSConfig()
    mcts = MCTS(seed, adv_eq, gym_config, mcts_config)
    
    print("Training")
    mcts.train(100)
    
    print("Evaluating")
    all_episode_rewards, best_rewards, optimal_states_all = mcts.evaluate()
    res = {"all_episode_rewards": all_episode_rewards, "best_rewards": best_rewards, "optimal_states_all": optimal_states_all}
    scipy.io.savemat("MCTS_results/mcts_results.mat", res)