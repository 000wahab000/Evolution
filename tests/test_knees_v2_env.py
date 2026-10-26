import math
import random

import pytest

from src.environment import CreatureEnv

RANGES = {True: (-90.0, 0.0), "spot": (-150.0, -14.0)}


def knee_deg(env):
    return [math.degrees(s.angle - t.angle) for t, s in zip(env.legs, env.shins)]


def test_default_env_unchanged():
    env = CreatureEnv()
    assert len(env.reset()) == 15
    assert len(env.legs) == 4 and len(env.motors) == 4


def test_invalid_args_raise():
    with pytest.raises(ValueError):
        CreatureEnv(knees="foo")
    with pytest.raises(ValueError):
        CreatureEnv(fall_rule="foo")


def test_penalty_rule_costs_two_and_does_not_fall():
    results = {}
    for rule in ("touch", "penalty"):
        env = CreatureEnv(fall_rule=rule)
        env.reset()
        env.torso.position = (400, 490)  # torso bottom (500) is below the ground top (495)
        _, reward, _, fallen = env.step([0] * 4)
        results[rule] = (reward, fallen)
    assert results["touch"][1] is True
    assert results["penalty"][1] is False
    assert results["penalty"][0] == pytest.approx(results["touch"][0] - 2.0, abs=0.5)


def test_penalty_rule_tilt_still_falls():
    env = CreatureEnv(fall_rule="penalty")
    env.reset()
    env.torso.angle = 1.3
    _, _, _, fallen = env.step([0] * 4)
    assert fallen is True


@pytest.mark.parametrize("knees", [True, "spot"])
def test_knee_stays_within_limit(knees):
    lo, hi = RANGES[knees]
    rnd = random.Random(0)
    plans = [
        lambda: [0] * 4 + [1] * 4,
        lambda: [0] * 4 + [-1] * 4,
        lambda: [rnd.uniform(-1, 1) for _ in range(8)],
    ]
    for plan in plans:
        env = CreatureEnv(knees=knees, fall_rule="penalty")
        env.reset()
        for _ in range(60):
            env.step(plan())
            assert all(lo - 5 <= k <= hi + 5 for k in knee_deg(env))


def test_spot_reset_pose_and_stands():
    env = CreatureEnv(knees="spot", fall_rule="penalty")
    state = env.reset()
    assert len(state) == 23 and len(env.motors) == 8
    assert all(k == pytest.approx(-30.0, abs=1e-6) for k in knee_deg(env))
    for _ in range(30):
        _, _, _, fallen = env.step([0] * 8)
        assert fallen is False
