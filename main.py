"""
This file trains a model using the HorizontalCREnv-V0 environment
"""

import gymnasium as gym
from stable_baselines3 import PPO
import numpy as np

import bluesky_gym
import bluesky_gym.envs

bluesky_gym.register_envs()

TRAIN = False
EVAL_EPISODES = 10
EPOCHS = 200

if __name__ == "__main__":
    # Create the environment
    env = gym.make('VerticalCREnv-v0', render_mode=None)
    obs, info = env.reset()

    # Create the model
    model = PPO("MultiInputPolicy", env, verbose=1,learning_rate=3e-4)

    # Train the model
    if TRAIN:
        for i in range(EPOCHS):
            model.learn(total_timesteps=int(20e5/EPOCHS))
            model.save("models/VerticalCREnv-v0_ppo/model")
        del model
    
    env.close()
    
    # Test the trained model

    # model = PPO.load("models/HorizontalCREnv-v0_ppo/model", env=env)
    env = gym.make('VerticalCREnv-v0', render_mode="human")

    for i in range(EVAL_EPISODES):
        done = truncated = False
        obs, info = env.reset()
        tot_rew = 0
        while not (done or truncated):
            # Predict
            action = np.array(np.random.randint(-100,100)/1000)
            # action, _states = model.predict(obs, deterministic=True)
            # Get reward
            obs, reward, done, truncated, info = env.step(action[()])
            tot_rew += reward

        print(tot_rew)


    env.close()