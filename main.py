from src.OED import OEDGymConfig
from src.DQN import DQN_OED
from src.GA import GA_OED, GAConfig
from pde.AdvectionEquation import Advection2D, Adv2dModelConfig
from pde.Burgers2D import Burgers2D, Burgers2DConfig
from pde.AdvectionDiffusionReaction import ADR, ADRConfig

import scipy.io
import os


def main(system_name, run_number, seed, cases):
    if system_name == "Advection2D":
        pde_system = Advection2D(Adv2dModelConfig())
        gym_config = OEDGymConfig()
        gym_config.n_sensor = 2
        gym_config.n_components_rewards = 0.99
        
    elif system_name == "Burgers2D":
        pde_system = Burgers2D(Burgers2DConfig())
        gym_config = OEDGymConfig()
        gym_config.n_sensor = 4
        gym_config.n_components_rewards = 4
        
    elif system_name == "ADR":
        pde_system = ADR(ADRConfig())
        gym_config = OEDGymConfig()
        gym_config.n_sensor = 5
        gym_config.n_components_rewards = 5
        
    os.makedirs(f"Plots/{system_name}_{run_number}_{seed}", exist_ok=False)
        
    if "old_dqn" in cases:
        print("Old DQN")
        gym_config.old_action_space = True
        dqn_old = DQN_OED(seed, pde_system, gym_config, verbose=1)
        
        model_name = f"{system_name}_old_dqn_{run_number}_seed_{seed}"
        dqn_old.train(model_name, total_timesteps=50000)
        print(f"training complete, saved at {model_name}")
        
        
    if "new_dqn" in cases:
        print("New DQN")
        gym_config.old_action_space = False
        dqn_new = DQN_OED(seed, pde_system, gym_config, verbose=0)
        
        model_name = f"{system_name}_new_dqn_{run_number}_seed_{seed}"
        dqn_new.train(model_name, total_timesteps=50000)
        print(f"training complete, saved at {model_name}")
        
    if "ga" in cases:
        print("GA")
        ga_config = GAConfig()
        ga = GA_OED(seed, pde_system, gym_config, ga_config)
        
        model_name = f"{system_name}_ga_{run_number}_seed_{seed}"
        best_individual, best_fitness, sensor_positions, logbook = ga.run()
        scipy.io.savemat(model_name + ".mat", {
            "best_individual": best_individual,
            "best_fitness": best_fitness,
            "sensor_positions": sensor_positions,
            "logbook": logbook
        })
                
        print(f"training complete, saved at {model_name}")
        
        