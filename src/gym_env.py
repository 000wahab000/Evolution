"""Gymnasium wrapper around CreatureEnv (optional: pip install -e .[gym])."""

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from src.environment import CreatureEnv


class QuadrupedGymEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, max_episode_steps=1200, cpg_mode="cpg", fitness_mode="baseline"):
        self.max_episode_steps = max_episode_steps
        self.cpg_mode, self.fitness_mode = cpg_mode, fitness_mode
        self.observation_space = spaces.Box(-np.inf, np.inf, (15,), np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, (4,), np.float32)
        self.env = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # fresh physics world each episode; the noise CPG mode draws from the seeded np_random
        self.env = CreatureEnv(fitness_mode=self.fitness_mode, cpg_mode=self.cpg_mode,
                               cpg_rng=self.np_random)
        self.t = 0
        return self.env.reset(), {"distance": 0.0, "fallen": False}

    def step(self, action):
        state, reward, done, fallen = self.env.step(np.clip(action, -1.0, 1.0))
        self.t += 1
        info = {"distance": float(self.env.torso.position.x - 400.0), "fallen": bool(fallen)}
        return state, float(reward), bool(done), self.t >= self.max_episode_steps and not done, info


gym.register(id="Evolution1-Quadruped-v0", entry_point="src.gym_env:QuadrupedGymEnv")
