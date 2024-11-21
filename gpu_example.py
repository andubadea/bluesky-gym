"""
This file is an example train and test loop for the different environments that
uses the GPU for training.
Note that using the GPU for training does not always result in faster computation.
This is highly dependent on the environment and algorithm combination. For
simple environments, the extra overhead of copying information to/from the
GPU results in slower performance in comparison with only using the CPU.
Selecting different environments is done through setting the 'env_name' variable.
"""

import torch as th
import stable_baselines3 as sb3
from typing import Dict, Union
import numpy as np

# Check if mps or cuda are available.
device = "mps" if th.backends.mps.is_built() \
    else "cuda" if th.cuda.is_available() else "cpu"
    
if device == "cpu":
    print('A compatible GPU was not detected. Defaulting to using CPU.')
else:
    print(f'Using the following GPU device: {device}.')
    
if device =="mps":
    # Stable baselines 3, as of right now, does not support MPS. But with a
    # small tweak it can. We need to overwrite some functions to make tensors
    # use float32 as the data type.
    
    def mps_obs_as_tensor(obs: Union[np.ndarray, Dict[str, np.ndarray]], 
                          device: th.device) -> Union[th.Tensor, 
                            sb3.common.type_aliases.TensorDict]:
        dtype = th.float32
        if isinstance(obs, np.ndarray):
            return th.as_tensor(obs, device=device, dtype=dtype)
        elif isinstance(obs, dict):
            return {key: th.as_tensor(_obs, device=device, dtype=dtype) 
                    for (key, _obs) in obs.items()}
        else:
            raise Exception(f"Unrecognized type of observation {type(obs)}")
        
    def mps_to_torch(self, array: np.ndarray, copy: bool = True) -> th.Tensor:
        """
        Convert a numpy array to a PyTorch tensor.
        Note: it copies the data by default

        :param array:
        :param copy: Whether to copy or not the data (may be useful to avoid changing things
            by reference). This argument is inoperative if the device is not the CPU.
        :return:
        """
        dtype = th.float32
        if copy:
            return th.tensor(array, device=self.device, dtype=dtype)
        return th.as_tensor(array, device=self.device, dtype=dtype)
    
    # Now overwrite the functions
    sb3.common.utils.obs_as_tensor.__code__ = mps_obs_as_tensor.__code__
    sb3.common.buffers.BaseBuffer.to_torch.__code__ = mps_to_torch.__code__
    # And now import the algorithms from the new class
    PPO = sb3.PPO
    SAC = sb3.SAC
    TD3 = sb3.TD3
    DDPG = sb3.DDPG

else:
    # Import the usual stable_baselines3 algorithms
    from stable_baselines3 import PPO, SAC, TD3, DDPG

import gymnasium as gym

import bluesky_gym
import bluesky_gym.envs

from bluesky_gym.utils import logger

bluesky_gym.register_envs()

env_name = 'SectorCREnv-v0'
algorithm = SAC

# Initialize logger
log_dir = f'./logs/{env_name}/'
file_name = f'{env_name}_{str(algorithm.__name__)}.csv'
csv_logger_callback = logger.CSVLoggerCallback(log_dir, file_name)

TRAIN = True
EVAL_EPISODES = 10

if __name__ == "__main__":
    env = gym.make(env_name, render_mode=None)
    obs, info = env.reset()
    model = algorithm("MultiInputPolicy", 
                      env, 
                      verbose=1,
                      learning_rate=3e-4, 
                      device=device)
    if TRAIN:
        model.learn(total_timesteps=2e6, callback=csv_logger_callback)
        model.save(f"models/{env_name}/{env_name}_{str(algorithm.__name__)}/model_gpu")
        del model
    env.close()
    
    # Test the trained model
    model = algorithm.load(f"models/{env_name}/{env_name}_{str(algorithm.__name__)}/model_gpu", env=env)
    env = gym.make(env_name, render_mode="human")
    for i in range(EVAL_EPISODES):

        done = truncated = False
        obs, info = env.reset()
        tot_rew = 0
        while not (done or truncated):
            # action = np.array(np.random.randint(-100,100,size=(2))/100)
            # action = np.array([0,-1])
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action[()])
            tot_rew += reward
        print(tot_rew)
    env.close()