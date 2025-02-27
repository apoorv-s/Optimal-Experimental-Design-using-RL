import numpy as np
from GymEnvironment.environment import SensorOptimalPlacement


def run_random_policy_benchmark(
        width=50,
        length=50,
        n_sensor=5,
        seed=42,
        num_episodes=10,
        max_horizon = 20,
        N_max = 5,
):
    # Create an instance of your custom env
    env = SensorOptimalPlacement(width=width, length=length, n_sensor=n_sensor, max_horizon= max_horizon, N_max= N_max, seed=seed)

    all_episode_rewards = []

    for episode_idx in range(num_episodes):
        # Reset environment and get initial observation
        obs = env.reset(seed = episode_idx)

        done = False

        while not done:
            # Sample a random action
            action = env.action_space.sample()

            # Take a step in the environment
            obs, reward, done, info = env.step(action)

            # Check for a terminal condition (if your env supports it)
            if done:
                break

        # Store this episode's reward
        # all_episode_rewards.append(episode_reward)
        # print(f"Episode {episode_idx + 1}/{num_episodes} - Reward: {episode_reward}")

    # Print or return some statistics
    avg_reward = np.mean(all_episode_rewards)
    print(f"\nCompleted {num_episodes} episodes using random policy.")
    print(f"Average reward: {avg_reward:.4f}")
    print(f"Reward per episode: {all_episode_rewards}")

    return all_episode_rewards


if __name__ == "__main__":
    # Run the benchmark with random actions
    run_random_policy_benchmark()
