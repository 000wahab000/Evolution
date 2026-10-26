import math
from dataclasses import asdict

import numpy as np
import pytest

from src.environment import STUCK_WINDOW, CreatureEnv, run_episode
from src.research.config import RunConfig, config_sha256

BODIES = {"straight": {}, "knees": {"knees": True}, "spot": {"knees": "spot"}}


def states(env, n=40, seed=3):
    rng = np.random.default_rng(seed)
    out = [env.reset()]
    for _ in range(n):
        s, r, d, f = env.step(rng.uniform(-1, 1, len(env.motors)))
        out.append((s, r, d, f))
    return out


@pytest.mark.parametrize("kw", [{}, {"knees": True}, {"knees": "spot", "fall_rule": "penalty"}])
def test_guards_off_matches_default_env(kw):
    a, b = CreatureEnv(**kw), CreatureEnv(guards=False, **kw)
    sa, sb = states(a), states(b)
    assert np.array_equal(sa[0], sb[0])
    for (s1, r1, d1, f1), (s2, r2, d2, f2) in zip(sa[1:], sb[1:]):
        assert np.array_equal(s1, s2) and (r1, d1, f1) == (r2, d2, f2)
    assert not b.invalid


@pytest.mark.parametrize("name", BODIES)
def test_hips_stay_within_limit_with_guards(name):
    env = CreatureEnv(guards=True, fall_rule="penalty", **BODIES[name])
    limit = 90.0 if name == "spot" else 60.0
    rng = np.random.default_rng(0)
    env.reset()
    worst = 0.0
    for _ in range(100):
        env.step(rng.uniform(-1, 1, len(env.motors)))
        worst = max(worst, max(abs(math.degrees(leg.angle - env.torso.angle)) for leg in env.legs))
        if env.invalid:
            break
    assert worst <= limit + 5.0


def test_huge_velocity_is_invalid_without_penalty():
    env = CreatureEnv(guards=True, fall_rule="penalty")
    env.reset()
    env.step(np.zeros(4))
    sane_x = env.score_x
    env.torso.velocity = (5000.0, 0.0)
    _, reward, done, fallen = env.step(np.zeros(4))
    assert env.invalid and done and not fallen and reward == 0.0
    assert env.torso.position.x > sane_x + 50  # the bad step moved far, but...
    assert env.score_x == sane_x  # ...the score stays at the last sane position


def test_invalid_episode_has_no_fall_penalty():
    env = CreatureEnv(guards=True, fall_rule="penalty")

    class Brain:
        def get_action(self, state):
            env.torso.angular_velocity = 500.0  # run_episode resets the env, so spin it here
            return np.zeros(4)

    ret, steps, fallen = run_episode(env, Brain(), 50)
    assert env.invalid and not fallen and steps == 1 and ret == 0.0


@pytest.mark.parametrize("name", BODIES)
def test_standing_still_ends_as_stuck(name):
    env = CreatureEnv(guards=True, fall_rule="penalty", **BODIES[name])
    env.reset()
    steps, done, fallen = 0, False, False
    while not done and steps < 600:
        _, _, done, fallen = env.step(np.zeros(len(env.motors)))
        steps += 1
    assert done and not fallen and not env.invalid
    assert steps <= STUCK_WINDOW + 5


def test_shuffling_ends_as_stuck():
    env = CreatureEnv(guards=True, fall_rule="penalty", knees="spot")
    env.reset()
    steps, done = 0, False
    while not done and steps < 600:
        a = np.zeros(8)
        a[0] = 0.3 * math.sin(steps / 3)  # small shuffle, far less than the window threshold
        _, _, done, _ = env.step(a)
        steps += 1
    assert done and not env.invalid and steps <= STUCK_WINDOW + 5


def test_config_hash_unchanged_at_default_and_differs_with_guards():
    cfg = RunConfig("e", "v", 0)
    d = asdict(cfg)
    for k in ("knees", "fall_rule", "guards"):
        del d[k]
    assert cfg.sha256() == config_sha256(d)
    assert RunConfig("e", "v", 0, guards=True).sha256() != cfg.sha256()
