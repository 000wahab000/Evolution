import pytest

gymnasium = pytest.importorskip("gymnasium")
from gymnasium.utils.env_checker import check_env  # noqa: E402

import src  # noqa: E402,F401  (registers the env)


def test_check_env():
    check_env(gymnasium.make("Evolution1-Quadruped-v0").unwrapped)


def test_random_rollout():
    env = gymnasium.make("Evolution1-Quadruped-v0")
    obs, _ = env.reset(seed=0)
    assert obs.shape == (15,)
    for _ in range(50):
        obs, r, term, trunc, info = env.step(env.action_space.sample())
        if term or trunc:
            break
    assert "distance" in info and "fallen" in info
