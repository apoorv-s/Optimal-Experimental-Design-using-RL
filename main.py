from src.OED import OEDGymConfig
from src.DQN import DQN_OED, DQNConfig
from src.GA import GA_OED, GAConfig
from pde.AdvectionEquation import Advection2D, Adv2dModelConfig
from pde.Burgers2D import Burgers2D, Burgers2DConfig
from pde.AdvectionDiffusionReaction import ADR, ADRConfig

import scipy.io
import os
import torch.nn as nn
from argparse import ArgumentParser


def run_advection2d(run_number, seed, cases):
    pde_system = Advection2D(Adv2dModelConfig())
    gym_config = OEDGymConfig()
    gym_config.n_sensor = 2
    gym_config.n_components_rewards = 0.99
    
    dqn_config = DQNConfig()
    
    if "old_dqn" in cases:
        print("Old DQN")
        
        gym_config.old_action_space = True
        gym_config.max_horizon = 5
        
        dqn_config.net_arch = [64, 128, 256, 512, 1024]
        dqn_config.activation_fn = nn.Tanh
        dqn_config.learning_rate = 2e-6
        
        dqn_old = DQN_OED(seed, pde_system, gym_config, dqn_config, verbose=1)
        
        model_name = f"Advection2D_old_dqn_{run_number}_seed_{seed}"
        dqn_old.train(model_name, total_timesteps=30000)
        print(f"training complete, saved at {model_name}")
        
        log_dir = dqn_old.model.logger.dir
        print(f"Tensorboard logs are being saved to: {log_dir}")
        
    if "new_dqn" in cases:
        print("New DQN")
        gym_config.old_action_space = False
        gym_config.max_horizon = 500
        
        dqn_config.net_arch = [64, 128, 256, 128, 64, 32]
        dqn_config.activation_fn = nn.Tanh
        dqn_config.learning_rate = 2e-6
        
        dqn_new = DQN_OED(seed, pde_system, gym_config, dqn_config, verbose=1)
        
        model_name = f"Advection2D_new_dqn_{run_number}_seed_{seed}"
        dqn_new.train(model_name, total_timesteps=200000)
        print(f"training complete, saved at {model_name}")
        
        log_dir = dqn_new.model.logger.dir
        print(f"Tensorboard logs are being saved to: {log_dir}")
    
    if "ga" in cases:
        print("GA")
        ga_config = GAConfig()
        ga = GA_OED(pde_system, gym_config, ga_config)
        
        model_name = f"GAResults/Advection2D_ga_{run_number}"
        best_individual, best_fitness, sensor_positions, logbook = ga.run()
        scipy.io.savemat(model_name + ".mat", {
            "best_individual": best_individual,
            "best_fitness": best_fitness,
            "sensor_positions": sensor_positions,
            "logbook": logbook
        })
                
        print(f"training complete, saved at {model_name}")
        
def run_burgers2d(run_number, seed, cases):
    pde_system = Burgers2D(Burgers2DConfig())
    gym_config = OEDGymConfig()
    gym_config.n_sensor = 4
    gym_config.n_components_rewards = 4
    
    dqn_config = DQNConfig()
    
    if "old_dqn" in cases:
        print("Old DQN")
        
        gym_config.old_action_space = True
        gym_config.max_horizon = 5
        
        dqn_config.net_arch = [64, 128, 256, 512, 1024]
        dqn_config.activation_fn = nn.Tanh
        dqn_config.learning_rate = 2e-6
        
        dqn_old = DQN_OED(seed, pde_system, gym_config, dqn_config, verbose=1)
        
        model_name = f"Burgers2D_old_dqn_{run_number}_seed_{seed}"
        dqn_old.train(model_name, total_timesteps=30000)
        print(f"training complete, saved at {model_name}")
        
        log_dir = dqn_old.model.logger.dir
        print(f"Tensorboard logs are being saved to: {log_dir}")
        
    if "new_dqn" in cases:
        print("New DQN")
        gym_config.old_action_space = False
        gym_config.max_horizon = 500
        
        dqn_config.net_arch = [64, 128, 256, 128, 64, 32]
        dqn_config.activation_fn = nn.Tanh
        dqn_config.learning_rate = 2e-6
        
        dqn_new = DQN_OED(seed, pde_system, gym_config, dqn_config, verbose=1)
        
        model_name = f"Burgers2D_new_dqn_{run_number}_seed_{seed}"
        dqn_new.train(model_name, total_timesteps=200000)
        print(f"training complete, saved at {model_name}")
        
        log_dir = dqn_new.model.logger.dir
        print(f"Tensorboard logs are being saved to: {log_dir}")
    
    if "ga" in cases:
        print("GA")
        ga_config = GAConfig()
        ga = GA_OED(pde_system, gym_config, ga_config)
        
        model_name = f"GAResults/Burgers2D_ga_{run_number}"
        best_individual, best_fitness, sensor_positions, logbook = ga.run()
        scipy.io.savemat(model_name + ".mat", {
            "best_individual": best_individual,
            "best_fitness": best_fitness,
            "sensor_positions": sensor_positions,
            "logbook": logbook
        })
        
        print(f"training complete, saved at {model_name}")
        
def run_adr(run_number, seed, cases):
    pass


# def run(system_name, run_number, seed, cases):
#     if system_name == "Advection2D":
#         pde_system = Advection2D(Adv2dModelConfig())
#         gym_config = OEDGymConfig()
#         gym_config.n_sensor = 2
#         gym_config.n_components_rewards = 0.99
        
#         dqn_config = DQNConfig()
#         dqn_config.net_arch = [64, 128, 256, 512, 1024]
#         dqn_config.activation_fn = nn.Tanh
#         dqn_config.learning_rate = 2e-6
        
        
#     elif system_name == "Burgers2D":
#         pde_system = Burgers2D(Burgers2DConfig())
#         gym_config = OEDGymConfig()
#         gym_config.n_sensor = 4
#         gym_config.n_components_rewards = 4
        
#         dqn_config = DQNConfig()
        
#     elif system_name == "ADR":
#         pde_system = ADR(ADRConfig())
#         gym_config = OEDGymConfig()
#         gym_config.n_sensor = 5
#         gym_config.n_components_rewards = 5
        
#         dqn_config = DQNConfig()
        
#     # os.makedirs(f"Plots/{system_name}_{run_number}_{seed}", exist_ok=False)
        
#     if "old_dqn" in cases:
#         print("Old DQN")
#         gym_config.old_action_space = True
#         gym_config.max_horizon = 5
        
#         dqn_old = DQN_OED(seed, pde_system, gym_config, dqn_config, verbose=1)
        
#         model_name = f"{system_name}_old_dqn_{run_number}_seed_{seed}"
#         dqn_old.train(model_name, total_timesteps=30000)
#         print(f"training complete, saved at {model_name}")
        
#         log_dir = dqn_old.model.logger.dir
#         print(f"Tensorboard logs are being saved to: {log_dir}")
           
#     if "new_dqn" in cases:
#         print("New DQN")
#         gym_config.old_action_space = False
#         gym_config.max_horizon = 1000
        
        
#         dqn_new = DQN_OED(seed, pde_system, gym_config, dqn_config, verbose=1)
        
#         model_name = f"{system_name}_new_dqn_{run_number}_seed_{seed}"
#         dqn_new.train(model_name, total_timesteps=200000)
#         print(f"training complete, saved at {model_name}")
        
#         log_dir = dqn_new.model.logger.dir
#         print(f"Tensorboard logs are being saved to: {log_dir}")
        
#     if "ga" in cases:
#         print("GA")
#         ga_config = GAConfig()
#         ga = GA_OED(seed, pde_system, gym_config, ga_config)
        
#         model_name = f"{system_name}_ga_{run_number}_seed_{seed}"
#         best_individual, best_fitness, sensor_positions, logbook = ga.run()
#         scipy.io.savemat(model_name + ".mat", {
#             "best_individual": best_individual,
#             "best_fitness": best_fitness,
#             "sensor_positions": sensor_positions,
#             "logbook": logbook
#         })
                
#         print(f"training complete, saved at {model_name}")
        
        
if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--sys_name", type=str, required=True)
    parser.add_argument("--run_num", type=int, required=True)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--cases", type=str, required=True)
    args = parser.parse_args()
    
    if args.sys_name == "Advection2D":
        run_advection2d(args.run_num, args.seed, args.cases.split())
    elif args.sys_name == "Burgers2D":
        pass
    elif args.sys_name == "ADR":
        pass
    else:
        print("Invalid system name")
        exit()