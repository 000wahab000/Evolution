import numpy as np
import pymunk

from src.environment import CreatureEnv
from src.model import CreatureBrain
from src.terrain import fixed_course, legacy_course, random_course
from src.utils import UNIVERSAL_BEST_PATH, load_state_dict_file


def old_ground(space):
    """Verbatim geometry of the pre-TerrainSpec create_ground() + terrain_changing()."""
    base = pymunk.Segment(space.static_body, (-10000, 500), (3000, 500), 5.0)
    base.friction = 1.2
    space.add(base)
    for start_x, end_x in ((600, 1600), (1600, 2600)):
        body = pymunk.Body(body_type=pymunk.Body.STATIC)
        seg = pymunk.Segment(body, (start_x, 500), (end_x, 500 - (end_x - start_x) / 25), 4)
        seg.friction = 1.8
        space.add(body, seg)


class OldEnv(CreatureEnv):
    def create_ground(self):
        old_ground(self.space)


def segs(space):
    return [(tuple(s.a), tuple(s.b), s.radius, s.friction)
            for s in space.shapes if isinstance(s, pymunk.Segment)]


def walk(env, brain, steps=600):
    state, done, n = env.reset(), False, 0
    while not done and n < steps:
        state, _, done, _ = env.step(brain.get_action(state))
        n += 1
    return env.torso.position.x


def test_legacy_course_matches_old_geometry():
    assert segs(CreatureEnv().space) == segs(OldEnv().space)


def test_universal_best_walks_identically():
    brain = CreatureBrain()
    brain.load_state_dict(load_state_dict_file(UNIVERSAL_BEST_PATH))
    assert walk(CreatureEnv(), brain) == walk(OldEnv(), brain)


def test_fixed_course_is_continuous_and_goes_downhill():
    s = fixed_course().segments
    for a, b in zip(s, s[1:]):
        assert np.hypot(a.b[0] - b.a[0], a.b[1] - b.a[1]) <= 1.0
    assert s[-1].b == (3000, 500)
    assert fixed_course().ground_top(2100) > fixed_course().ground_top(1600)  # y down: lower ground


def test_random_course_deterministic_and_bounded():
    a = random_course(np.random.default_rng(5), 0.04)
    assert a == random_course(np.random.default_rng(5), 0.04)
    assert a != random_course(np.random.default_rng(6), 0.04)
    for s in a.segments[1:]:
        assert abs((s.b[1] - s.a[1]) / (s.b[0] - s.a[0])) <= 0.04 + 1e-12
    steep = random_course(np.random.default_rng(5), 0.08, min_slope=0.05)
    for s in steep.segments[1:]:
        assert 0.05 - 1e-12 <= abs((s.b[1] - s.a[1]) / (s.b[0] - s.a[0])) <= 0.08 + 1e-12


def test_non_legacy_fallen_check_follows_the_ground():
    course = random_course(np.random.default_rng(1), 0.04)
    env = CreatureEnv(terrain=course)
    env.reset()
    _, _, done, fallen = env.step([0, 0, 0, 0])
    assert not fallen  # standing at spawn
    env.torso.position = (2000, course.ground_top(2000) - 2)  # torso body pressed into the slope
    _, _, done, fallen = env.step([0, 0, 0, 0])
    assert done and fallen
