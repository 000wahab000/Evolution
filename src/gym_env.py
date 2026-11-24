"""Gymnasium wrapper around CreatureEnv (optional: pip install -e .[gym]).

The RL contract differs from the GA one in two ways, and both matter:

- `terminated` is a real terminal state (fall or stuck). Hitting the step budget is
  `truncated`, so PPO bootstraps the value of the remaining episode instead of treating the
  time limit as the end of the world.
- `fall_penalty` (default 100) is returned on the falling step. The GA applies the same -100
  outside the env in `run_episode`; an RL agent that never sees it learns to dive.
"""

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from src.environment import CreatureEnv

OBS_DIM = {False: 15, True: 23}
ACT_DIM = {False: 4, True: 8}


class QuadrupedGymEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, max_episode_steps=600, cpg_mode="cpg", fitness_mode="baseline",
                 fall_penalty=100.0, fall_rule="touch", knees=False, guards=False,
                 substeps=1, terrain=None, cpg_rng=None):
        self.max_episode_steps = max_episode_steps
        self.cpg_mode, self.fitness_mode = cpg_mode, fitness_mode
        self.fall_penalty, self.fall_rule = fall_penalty, fall_rule
        self.knees, self.guards, self.substeps = knees, guards, substeps
        self.terrain, self.cpg_rng = terrain, cpg_rng
        kn = bool(knees)
        self.observation_space = spaces.Box(-np.inf, np.inf, (OBS_DIM[kn],), np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, (ACT_DIM[kn],), np.float32)
        self.env = None
        self.t = 0

    def _make_env(self, cpg_rng):
        return CreatureEnv(fitness_mode=self.fitness_mode, cpg_mode=self.cpg_mode,
                           cpg_rng=cpg_rng, terrain=self.terrain, fall_rule=self.fall_rule,
                           knees=self.knees, guards=self.guards, substeps=self.substeps,
                           max_steps=self.max_episode_steps, fall_penalty=self.fall_penalty)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # fresh physics world each episode; the noise CPG mode draws from the seeded np_random
        self.env = self._make_env(self.cpg_rng if self.cpg_rng is not None else self.np_random)
        self.t = 0
        return self.env.reset(), {"distance": 0.0, "fallen": False}

    def step(self, action):
        action = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        state, reward, done, fallen = self.env.step(action)
        self.t += 1
        truncated = bool(self.env.truncated)
        terminated = bool(done and not truncated)
        info = {"distance": float(self.env.score_x - 400.0), "fallen": bool(fallen),
                "steps": self.t}
        return state, float(reward), terminated, truncated, info


gym.register(id="Evolution1-Quadruped-v0", entry_point="src.gym_env:QuadrupedGymEnv")
