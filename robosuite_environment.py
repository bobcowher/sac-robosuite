import robosuite as suite
import numpy as np
import torch
from gym import spaces

from robosuite.wrappers import Wrapper

class RoboSuiteWrapper(Wrapper):

    def __init__(self, env):

        super().__init__(env)

        self.max_episode_steps = 300
        self.current_episode_step = 0
        self.max_action = 1
        self.input_dims = 84

    def step(self, action):
        observation, reward, done, info = super().step(action)
        observation = self.observation_to_tensor(observation)

        # Increment timesteps and set done if max timesteps reached
        self.current_episode_step += 1

        if self.current_episode_step == self.max_episode_steps:
            done = True

        return observation, reward, done, info

    def reset(self):
        self.current_episode_step = 0
        observation = super().reset()
        observation = self.observation_to_tensor(observation)
        return observation


    def observation_to_tensor(self, obs):
        # Convert each array in the ordered dictionary to a flattened numpy array
        flattened_arrays = [np.array(item).flatten() for item in obs.values()]

        # Concatenate all the flattened arrays to get a single array
        concatenated_array = np.concatenate(flattened_arrays)

        # Convert the numpy array to a PyTorch tensor
        return torch.tensor(concatenated_array, dtype=torch.float32)
