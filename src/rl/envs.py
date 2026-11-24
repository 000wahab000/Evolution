"""Gymnasium env on the research train distribution (needs the `rl` extra).

Every reset draws a fresh scenario with `sample_scenario`, the same function the GA research
runner uses, so PPO and the GA see the same courses, friction range and starting-pose noise.
"""

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from src.research.evaluate import SPAWN_X, Scenario, make_env, sample_scenario, split_terrain

OBS_DIM = {False: 15, True: 23}
ACT_DIM = {False: 4, True: 8}


class ScenarioGymEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, cfg, rng, split="train", substeps=1, fall_penalty=100.0):
        self.cfg, self.rng, self.split = cfg, rng, split
        self.substeps, self.fall_penalty = substeps, fall_penalty
        kn = bool(cfg.knees)
        self.observation_space = spaces.Box(-np.inf, np.inf, (OBS_DIM[kn],), np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, (ACT_DIM[kn],), np.float32)
        self.env = None
        self.t = 0

    def _scenario(self):
        if self.split == "train":
            return sample_scenario(self.cfg, self.rng)
        seeds = getattr(self.cfg, f"{self.split}_terrain_seeds")
        seed = int(self.rng.choice(seeds))
        return Scenario(split_terrain(self.cfg, self.split, seed), cpg_seed=seed)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        scenario = self._scenario()
        self.env = make_env(self.cfg, scenario, self.substeps, fall_penalty=self.fall_penalty,
                            max_steps=self.cfg.max_steps)
        self.t = 0
        return self.env.reset(pose_noise=scenario.pose_noise), {}

    def step(self, action):
        action = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        state, reward, done, fallen = self.env.step(action)
        self.t += 1
        truncated = bool(self.env.truncated)
        terminated = bool(done and not truncated)
        info = {"distance": float(self.env.score_x - SPAWN_X), "fallen": bool(fallen),
                "energy": float(np.abs(action).sum()), "steps": self.t}
        return state, float(reward), terminated, truncated, info
