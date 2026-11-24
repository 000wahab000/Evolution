"""RL contract of the envs: terminated vs truncated, and the fall penalty PPO needs.

These do not need torch, only gymnasium.
"""

import numpy as np
import pytest

pytest.importorskip("gymnasium")

from src.environment import CreatureEnv  # noqa: E402
from src.gym_env import QuadrupedGymEnv  # noqa: E402


def test_time_limit_is_truncation_not_termination():
    env = CreatureEnv(max_steps=3, fall_penalty=0.0)
    env.reset()
    for _ in range(3):
        _, _, done, fallen = env.step(np.zeros(4, dtype=np.float32))
    assert done and env.truncated and not fallen


def test_fall_penalty_is_returned_by_the_env():
    penalized = CreatureEnv(max_steps=10, fall_penalty=100.0)
    plain = CreatureEnv(max_steps=10, fall_penalty=0.0)
    for env in (penalized, plain):
        env.reset()
        env.torso.angle = 2.0  # > the 1.2 rad fall threshold
    _, r_pen, done_pen, fallen_pen = penalized.step(np.zeros(4, dtype=np.float32))
    _, r_plain, done_plain, fallen_plain = plain.step(np.zeros(4, dtype=np.float32))
    assert fallen_pen and fallen_plain and done_pen and done_plain
    assert r_plain > -10.0
    assert r_pen < -90.0
    assert abs((r_plain - r_pen) - 100.0) < 1e-6


def test_gym_env_splits_terminated_and_truncated():
    env = QuadrupedGymEnv(max_episode_steps=3)
    obs, _ = env.reset(seed=0)
    assert obs.shape == (15,)
    terminated = truncated = False
    for _ in range(3):
        obs, _, terminated, truncated, info = env.step(np.zeros(4, dtype=np.float32))
    assert truncated and not terminated
    assert info["steps"] == 3


def test_gym_env_terminates_on_fall_with_penalty():
    env = QuadrupedGymEnv(max_episode_steps=10, fall_penalty=100.0)
    env.reset(seed=0)
    env.env.torso.angle = 2.0
    _, reward, terminated, truncated, info = env.step(np.zeros(4, dtype=np.float32))
    assert terminated and not truncated and info["fallen"]
    assert reward < -90.0


def test_scenario_env_uses_the_research_distribution():
    from src.research.config import RunConfig
    from src.rl.envs import ScenarioGymEnv

    cfg = RunConfig(experiment="t", variant="v", seed=0, max_steps=5)
    env = ScenarioGymEnv(cfg, np.random.default_rng(0))
    obs, _ = env.reset(seed=0)
    assert obs.shape == (15,) and env.action_space.shape == (4,)
    _, _, _, truncated, _ = env.step(np.zeros(4, dtype=np.float32))
    assert not truncated

    knee_cfg = RunConfig(experiment="t", variant="v", seed=0, max_steps=5, knees=True)
    knee_env = ScenarioGymEnv(knee_cfg, np.random.default_rng(0))
    obs, _ = knee_env.reset(seed=0)
    assert obs.shape == (23,) and knee_env.action_space.shape == (8,)
