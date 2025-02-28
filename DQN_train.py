import gymnasium as gym
from stable_baselines3 import DQN
from GymEnvironment.environment import SensorOptimalPlacement
import numpy as np

if __name__ == "__main__":
    width = 50 #must match x in AdvectionEquation
    length = 50 #must match y in AdvectionEquation
    n_sensor = 5
    seed = 0
    num_episodes = 100
    max_horizon = 500
    N_max = 10
    # Create an instance of your custom env
    env = SensorOptimalPlacement(width=width, length=length, n_sensor=n_sensor, max_horizon=max_horizon, N_max=N_max,
                                 seed=seed)

    #DQN has a bunch of hyperparameter to be tuned, check: https://stable-baselines3.readthedocs.io/en/master/modules/dqn.html
    model = DQN("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=100000
                , log_interval=10)
    #here it means the agent will interact with the environment 10000 times, logging training process every 4 episodes, the env will auto reset when terminate

    #Evaluation
    #Note: Since we randomly init the environment, and each step we choose an optimal action, it can still take a while for the sensors to be moved to correct positions
    #So we would expect the outcome to depend on initial condition, so num_episode, max_horizon, N_max paramters all matters

    all_episode_rewards = []

    best_rewards = []
    for episode_idx in range(num_episodes):
        # Reset environment and get initial observation
        obs, _ = env.reset(seed=episode_idx)
        episode_rewards = []
        done = False
        truncated = False
        print(f"Starting episode {episode_idx + 1}/{num_episodes}")
        step = 0
        while not done and not truncated:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            episode_rewards.append(reward)
            step += 1
            print(f"  Step {step}, Current reward: {reward:.6f}")
        best_rewards.append(info["max_reward"])
        all_episode_rewards.append(episode_rewards)
        print(f"Episode {episode_idx + 1}/{num_episodes} complete - Max Reward: {info['max_reward']:.6f}")
    print(f"\nCompleted {num_episodes} episodes using DQN policy.")
    print(f"Best reward overall: {max(best_rewards):.6f}")
    print(f"Average best reward per episode: {np.mean(best_rewards):.6f}")