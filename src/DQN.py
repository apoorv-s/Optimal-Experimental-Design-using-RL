from stable_baselines3 import DQN
from src.OED import OED, OEDGymConfig
from tqdm import trange

import torch
import torch.nn as nn

class DQNConfig():
    def __init__(self):
        self.net_arch = [64, 128, 64]
        self.gamma = 1
        self.batch_size = 128
        self.learning_rate = 1e-5
        self.activation_fn = nn.LeakyReLU
        

class DQN_OED():
    def __init__(self, seed, pde_system, gym_config: OEDGymConfig, dqn_config: DQNConfig, verbose = 1):
        self.env = OED(pde_system, gym_config)
        self.gym_config = gym_config
        self.dqn_config = dqn_config
        
        # Can define these as inputs to the class
        self.model = DQN("MlpPolicy",
                         self.env,
                         verbose=verbose,
                         tensorboard_log= "./tensorboard/",
                         seed = seed,
                         gamma= dqn_config.gamma,
                         batch_size = dqn_config.batch_size,
                         learning_rate = dqn_config.learning_rate,
                         policy_kwargs=dict(
                             net_arch=dqn_config.net_arch,
                             activation_fn = dqn_config.activation_fn
                             ))
        
    def train(self, model_name, total_timesteps=50000, log_interval=10):
        self.model.learn(total_timesteps=total_timesteps, log_interval=log_interval)
        self.model.save("trained_model/" + model_name)
        self.tensorboard_log = self.model.logger.dir
        torch.save({'gym_config': self.gym_config, 'dqn_config': self.dqn_config, 'tensorboard_log': self.tensorboard_log}, f"trained_model/{model_name}_config.pt")
        
        
    def load(self, model_name):
        self.model = DQN.load(model_name)
        
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