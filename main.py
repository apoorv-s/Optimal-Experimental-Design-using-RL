import gymnasium as gym
from stable_baselines3 import DQN
from GymEnvironment.environment import SensorOptimalPlacement

if __name__ == "__main__":
    width = 50 #must match x in AdvectionEquation
    length = 50 #must match y in AdvectionEquation
    n_sensor = 5
    env = SensorOptimalPlacement(width, length, n_sensor)

    #DQN has a bunch of hyperparameter to be tuned, check: https://stable-baselines3.readthedocs.io/en/master/modules/dqn.html
    model = DQN("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=10000, log_interval=4)
    #here it means the agent will interact with the environment 10000 times, logging training process every 4 episodes, the env will auto reset when terminate

    #this part is probably for evaluation
    obs, info = env.reset()
    while True:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            obs, info = env.reset()