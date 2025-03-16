from stable_baselines3 import DQN
from src.OED import OED, OEDGymConfig
from tqdm import trange    


class DQN_OED():
    def __init__(self, seed, pde_system, gym_config: OEDGymConfig, verbose = 1):
        self.env = OED(pde_system, gym_config)
        
        # Can define these as inputs to the class
        self.model = DQN("MlpPolicy", self.env, verbose=verbose, tensorboard_log= "./tensorboard/", seed = seed,  policy_kwargs=dict(net_arch=[64, 128, 64]))
        
    def train(self, model_name, total_timesteps=50000, log_interval=10):
        self.model.learn(total_timesteps=total_timesteps, log_interval=log_interval)
        self.model.save("trained_model/" + model_name)
        
    def load(self, model_name):
        self.model = DQN.load("trained_model/" + model_name)
        
    def evaluate(self, num_episodes = 100):
        all_episode_rewards = []
        best_rewards = []
        optimal_states_all = []
        for episode_idx in trange(num_episodes):
            obs, _ = self.env.reset(seed=episode_idx)
            episode_rewards = []
            done = False
            truncated = False
            # print(f"Starting episode {episode_idx + 1}/{num_episodes}")
            step = 0
            while not done and not truncated:
                action, _states = self.model.predict(obs, deterministic=True)
                obs, reward, done, truncated, info = self.env.step(action)
                episode_rewards.append(reward)
                step += 1
                # print(f"  Step {step}, Current reward: {reward:.6f}")
            best_rewards.append(info["max_reward"])
            all_episode_rewards.append(episode_rewards)
            optimal_states_all.append(info["optimal_state"])
            # print(f"Episode {episode_idx + 1}/{num_episodes} complete - Max Reward: {info['max_reward']:.6f}")
        
        return all_episode_rewards, best_rewards, optimal_states_all
    
if __name__ == "__main__":
    import IPython
    IPython.embed()