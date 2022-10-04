import gym
from gym import spaces
from socnavenv.envs.socnavenv_v1 import SocNavEnv_v1
from socnavenv.envs.utils.wall import Wall
import numpy as np
import copy

class NoisyObservations(gym.Wrapper):
    def __init__(self, env: SocNavEnv_v1, mean, std_dev) -> None:
        """
        A Gaussian Noise of mean, and std_dev are added to the values of the observations that are received.
        """
        super().__init__(env)
        self.env = env
        self.mean = mean
        self.std_dev = std_dev

    def generate_random_noise(self):
        noise = np.random.randn()*self.std_dev + self.mean
        return noise

    def add_noise(self, obs):
        noisy_obs = obs
        noisy_obs["goal"][6] += self.generate_random_noise()
        noisy_obs["goal"][7] += self.generate_random_noise()
    
        for entity in ["humans", "tables", "laptops", "plants", "walls"]:
            o = noisy_obs[entity].reshape(-1, 13)
            for i in range(o.shape[0]):
                for j in range(6, 13): # noise is only added to the non-one-hot components of the observation
                    o[i][j] += self.generate_random_noise()
            noisy_obs[entity] = o.flatten()
        return noisy_obs

    def step(self, action_pre):
        obs, reward, done, info = self.env.step(action_pre)
        obs = self.add_noise(obs)
        return obs, reward, done, info

    def reset(self):
        obs = self.env.reset()
        obs = self.add_noise(obs)
        return obs

    def one_step_lookahead(self, action_pre):
        # storing a copy of env
        env_copy = copy.deepcopy(self.env)
        next_state, reward, done, info = env_copy.step(action_pre)
        next_state = self.add_noise(next_state)
        del env_copy
        return next_state, reward, done, info

    