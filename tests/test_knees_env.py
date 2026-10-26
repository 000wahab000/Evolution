import math

import pytest

from src.environment import CreatureEnv, run_episode
from src.model import CreatureBrain


def test_default_env_unchanged():
    env = CreatureEnv()
    assert len(env.reset()) == 15
    assert len(env.legs) == 4 and len(env.motors) == 4


def test_kneed_reset_shape():
    env = CreatureEnv(knees=True)
    state = env.reset()
    assert len(state) == 23
    assert len(env.legs) == 4 and len(env.shins) == 4 and len(env.motors) == 8
    assert all(state[i] == 0 for i in (15, 17, 19, 21))


def test_positive_knee_action_swings_foot_backward():
    env = CreatureEnv(knees=True)
    env.reset()
    for _ in range(30):
        state, *_ = env.step([0, 0, 0, 0, 1, 1, 1, 1])
    for i, (thigh, shin) in enumerate(zip(env.legs, env.shins)):
        rel = abs(state[15 + 2 * i])
        assert 0 < rel <= math.pi / 2 + 0.05
        assert shin.local_to_world((0, 10)).x < thigh.local_to_world((0, 10)).x


def test_kneed_zero_actions_stand():
    env = CreatureEnv(knees=True)
    env.reset()
    for _ in range(30):
        _, _, _, fallen = env.step([0] * 8)
        assert fallen is False


def test_kneed_pose_noise():
    env = CreatureEnv(knees=True)
    noise = [0.1, -0.1, 0.2, 0.0]
    state = env.reset(pose_noise=noise)
    for leg, th in zip(env.legs, noise):
        assert leg.angle == pytest.approx(th, abs=1e-6)
    assert all(abs(v) < 1e-6 for v in state[15::2])


def test_kneed_episode_runs():
    env = CreatureEnv(knees=True)
    brain = CreatureBrain(input_size=23, output_size=4 * 2)
    run_episode(env, brain, max_steps=100)
