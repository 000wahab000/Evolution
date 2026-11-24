"""The PPO policy must be exportable into CreatureBrain weights (needs the `rl` extra)."""

import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("stable_baselines3")

from gymnasium import spaces  # noqa: E402

from src.model import CreatureBrain  # noqa: E402
from src.rl.export import policy_to_state_dict  # noqa: E402
from src.rl.policy import TanhActorCriticPolicy  # noqa: E402


def _spaces():
    return (spaces.Box(-np.inf, np.inf, (15,), np.float32),
            spaces.Box(-1.0, 1.0, (4,), np.float32))


def _policy(**kwargs):
    obs_space, act_space = _spaces()
    return TanhActorCriticPolicy(obs_space, act_space, lr_schedule=lambda _: 3e-4, **kwargs)


def test_export_matches_the_policy_mean_action():
    policy = _policy()
    brain = CreatureBrain.from_state_dict(policy_to_state_dict(policy))
    rng = np.random.default_rng(0)
    for _ in range(20):
        obs = rng.normal(size=15).astype(np.float32)
        action, _ = policy.predict(obs, deterministic=True)
        assert np.allclose(action, brain.get_action(obs), atol=1e-5)


def test_export_keeps_the_creature_brain_shapes():
    sd = policy_to_state_dict(_policy())
    assert {k: v.shape for k, v in sd.items()} == {
        "w1": (16, 15), "b1": (16,), "w2": (4, 16), "b2": (4,)}
    assert all(v.dtype == np.float32 for v in sd.values())


def test_export_follows_the_hidden_width():
    sd = policy_to_state_dict(_policy(hidden_dim=64))
    assert sd["w1"].shape == (64, 15) and sd["w2"].shape == (4, 64)
