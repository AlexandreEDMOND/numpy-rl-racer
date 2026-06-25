import numpy as np


class ActionRepeatEnv:
    def __init__(self, env, skip_frames=4):
        if skip_frames < 1:
            raise ValueError(f"skip_frames must be >= 1, got {skip_frames}")
        self.env = env
        self.skip_frames = skip_frames

    def step(self, action):
        total_reward = np.float64(0.0)
        for _ in range(self.skip_frames):
            obs, reward, done, info = self.env.step(action)
            total_reward += reward
            if done:
                break
        return obs, total_reward, done, info

    def reset(self, seed=None):
        return self.env.reset(seed=seed)

    def __getattr__(self, name):
        return getattr(self.env, name)
