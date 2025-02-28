import numpy as np
from GymEnvironment.environment import SensorOptimalPlacement


def run_random_policy_benchmark(
        width=50,
        length=50,
        n_sensor=5,
        seed=42,
        num_episodes=100,
        max_horizon = 500,
        N_max = 10,
):
    # Create an instance of your custom env
    env = SensorOptimalPlacement(width=width, length=length, n_sensor=n_sensor, max_horizon= max_horizon, N_max= N_max, seed=seed)

    all_episode_rewards = []

    best_rewards = []
    for episode_idx in range(num_episodes):
        # Reset environment and get initial observation
        obs = env.reset(seed=episode_idx)
        episode_rewards = []
        done = False
        truncated = False
        print(f"Starting episode {episode_idx + 1}/{num_episodes}")
        step = 0
        while not done and not truncated:
            action = env.action_space.sample()
            obs, reward, done, truncated, info = env.step(action)
            episode_rewards.append(reward)
            step += 1
            print(f"  Step {step}, Current reward: {reward:.6f}")
        best_rewards.append(info["max_reward"])
        all_episode_rewards.append(episode_rewards)
        print(f"Episode {episode_idx + 1}/{num_episodes} complete - Max Reward: {info['max_reward']:.6f}")
    print(f"\nCompleted {num_episodes} episodes using random policy.")
    print(f"Best reward overall: {max(best_rewards):.6f}")
    print(f"Average best reward per episode: {np.mean(best_rewards):.6f}")

    return all_episode_rewards, best_rewards


if __name__ == "__main__":
    # Run the benchmark with random actions
    run_random_policy_benchmark()
