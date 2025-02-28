import gymnasium as gym
from stable_baselines3 import DQN
from GymEnvironment.environment import SensorOptimalPlacement
import numpy as np
import matplotlib.pyplot as plt

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
    model = DQN("MlpPolicy", env, verbose=1, tensorboard_log= "./tensorboard/")
    model.learn(total_timesteps=50000
                , log_interval=10)
    #to use tensorboard first do "conda install tensorboard"
    #then as the training is running, go to terminal and type "tensorboard --logdir ./tensorboard/", click on https://localhost:6006/
    #save and load model
    model.save("trained_model/trained_DQN")

    #I WOULD SUGGEST RUN THE TRAINING AND SAVE THE TRAINED MODEL SEPARATELY, THEN LOAD MODEL IF YOU WANT TO DO EVALUATION

    # model = DQN.load("trained_model/trained_DQN", env = env)
    #here it means the agent will interact with the environment 10000 times, logging training process every 4 episodes, the env will auto reset when terminate

    #Evaluation
    #Note: Since we randomly init the environment, and each step we choose an optimal action, it can still take a while for the sensors to be moved to correct positions
    #So we would expect the outcome to depend on initial condition, so num_episode, max_horizon, N_max paramters all matters

    all_episode_rewards = []

    best_rewards = []
    optimal_states_all = []
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
        optimal_states_all.append(info["optimal_state"])
        print(f"Episode {episode_idx + 1}/{num_episodes} complete - Max Reward: {info['max_reward']:.6f}")
    print(f"\nCompleted {num_episodes} episodes using DQN policy.")
    print(f"Best reward overall: {max(best_rewards):.6f}")
    print(f"Average best reward per episode: {np.mean(best_rewards):.6f}")

    # Plot distribution of best_rewards
    plt.figure(figsize=(8, 6))
    plt.hist(best_rewards, bins=20, color='skyblue', edgecolor='black', alpha=0.75)
    plt.xlabel('Episode Best Reward')
    plt.ylabel('Frequency')
    plt.title('Distribution of Episode Best Rewards')
    plt.grid(True)
    plt.show()

    # Compute sum of optimal_states over all episodes
    # Assumes that each info["optimal_states"] is a numpy array of the same shape
    optimal_states_stack = np.array(optimal_states_all)
    mean_optimal_states = np.sum(optimal_states_stack, axis=0)

    # Plot grid of mean optimal_states
    plt.figure(figsize=(8, 6))
    plt.imshow(mean_optimal_states, cmap='viridis', interpolation='nearest')
    plt.title('Total placements in optimal States over Episodes')
    plt.colorbar(label='Total placements')
    plt.xlabel('Grid X')
    plt.ylabel('Grid Y')
    plt.show()

