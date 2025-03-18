from src.OED import OEDGymConfig
from src.DQN import DQN_OED, DQNConfig
from src.GA import GA_OED, GAConfig
from pde.AdvectionEquation import Advection2D, Adv2dModelConfig
from pde.Burgers2D import Burgers2D, Burgers2DConfig
from pde.AdvectionDiffusionReaction import ADR, ADRConfig

import scipy.io
import matplotlib.pyplot as plt
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

        # evolution progress
        gen = logbook.select("gen")
        fit_max = logbook.select("max")
        fit_avg = logbook.select("avg")

        plt.figure(figsize=(10, 6))
        plt.plot(gen, fit_max, 'b-', label='Maximum Fitness')
        plt.plot(gen, fit_avg, 'r-', label='Average Fitness')
        plt.title('Evolution of Coverage')
        plt.xlabel('Generation')
        plt.ylabel('Coverage Percentage')
        plt.legend()
        plt.grid(True)
        plt.savefig(model_name + ".png")
        print(f"Best fitness: {best_fitness}")
        print(f"Sensor positions: {sensor_positions}")
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
        gym_config.max_horizon = 10
        
        dqn_config.net_arch = [64, 128, 256, 512, 1024, 2048]
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
        dqn_new.train(model_name, total_timesteps=150000)
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
        
        # evolution progress
        gen = logbook.select("gen")
        fit_max = logbook.select("max")
        fit_avg = logbook.select("avg")

        plt.figure(figsize=(10, 6))
        plt.plot(gen, fit_max, 'b-', label='Maximum Fitness')
        plt.plot(gen, fit_avg, 'r-', label='Average Fitness')
        plt.title('Evolution of Coverage')
        plt.xlabel('Generation')
        plt.ylabel('Coverage Percentage')
        plt.legend()
        plt.grid(True)
        plt.savefig(model_name + ".png")
        print(f"Best fitness: {best_fitness}")
        print(f"Sensor positions: {sensor_positions}")
        print(f"training complete, saved at {model_name}")
        
def run_adr(run_number, seed, cases):
    pde_system = ADR(ADRConfig())
    gym_config = OEDGymConfig()
    gym_config.n_sensor = 5
    gym_config.n_components_rewards = 5
    
    dqn_config = DQNConfig()
    
    if "old_dqn" in cases:
        print("Old DQN")
        
        gym_config.old_action_space = True
        gym_config.max_horizon = 20
        
        dqn_config.net_arch = [64, 128, 256, 512, 1024, 1024, 2048, 2048]
        dqn_config.activation_fn = nn.ReLU
        dqn_config.learning_rate = 2e-6
        
        dqn_old = DQN_OED(seed, pde_system, gym_config, dqn_config, verbose=1)
        
        model_name = f"ADR_old_dqn_{run_number}_seed_{seed}"
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
        
        model_name = f"ADR_new_dqn_{run_number}_seed_{seed}"
        dqn_new.train(model_name, total_timesteps=150000)
        print(f"training complete, saved at {model_name}")
        
        log_dir = dqn_new.model.logger.dir
        print(f"Tensorboard logs are being saved to: {log_dir}")
    
    if "ga" in cases:
        print("GA")
        ga_config = GAConfig()
        ga_config.generations = 2000
        ga = GA_OED(pde_system, gym_config, ga_config)
        
        model_name = f"GAResults/ADR_ga_{run_number}"
        best_individual, best_fitness, sensor_positions, logbook = ga.run()
        scipy.io.savemat(model_name + ".mat", {
            "best_individual": best_individual,
            "best_fitness": best_fitness,
            "sensor_positions": sensor_positions,
            "logbook": logbook
        })
        
        # evolution progress
        gen = logbook.select("gen")
        fit_max = logbook.select("max")
        fit_avg = logbook.select("avg")

        plt.figure(figsize=(10, 6))
        plt.plot(gen, fit_max, 'b-', label='Maximum Fitness')
        plt.plot(gen, fit_avg, 'r-', label='Average Fitness')
        plt.title('Evolution of reward')
        plt.xlabel('Generation')
        plt.ylabel('Reward')
        plt.legend()
        plt.grid(True)
        plt.savefig(model_name + ".png")
        print(f"Best fitness: {best_fitness}")
        print(f"Sensor positions: {sensor_positions}")
        print(f"training complete, saved at {model_name}")
        
        
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
        run_burgers2d(args.run_num, args.seed, args.cases.split())
    elif args.sys_name == "ADR2D":
        run_adr(args.run_num, args.seed, args.cases.split())
    else:
        print("Invalid system name")
        exit()