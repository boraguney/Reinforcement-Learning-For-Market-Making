import numpy as np

import gym
from gym import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(-np.array([1, 1, 1, 1]), np.array([1, 1, 1, 1]), dtype=np.float32)
        self.state = None

    def step(self, action):
        assert self.action_space.contains(action)
        
        self.state = (self.state[0] + action, self.state[1] + action, self.state[2] + action, self.state[3] + action)
        reward = np.linalg.norm(self.state)
        done = np.linalg.norm(self.state) < 0.1
        
        return np.array(self.state, dtype=np.float32), reward, done, False, {}

    def reset(self):
        self.state = self.np_random.uniform(-np.array([1, 1, 1, 1]), np.array([1, 1, 1, 1]), size=(4,))
        self.steps_beyond_terminated = None

        return np.array(self.state, dtype=np.float32), {}

gym.register(id='CustomEnv-v0', entry_point='environments.custom_env:CustomEnv')