import pytest

from src.environment import CreatureEnv


def test_environment_initialization():
    env = CreatureEnv(fitness_mode="baseline")
    assert env.steps == 0
    assert env.fitness_mode == "baseline"

def test_environment_multiple_resets():
    env = CreatureEnv()
    for _ in range(5):
        state = env.reset()
        assert len(state) == 15
        assert env.torso.position.x == 400
        assert env.torso.position.y == 450
        assert env.torso.angle == 0.0
        assert env.steps == 0


def test_environment_fitness_modes():
    for mode in ["baseline", "tilt_penalized", "energy_penalized", "all_penalties", "ground_termination"]:
        env = CreatureEnv(fitness_mode=mode)
        env.reset()
        action = [0.5, -0.5, 0.2, -0.2]
        state, reward, done, fallen = env.step(action)
        assert isinstance(reward, float)


def test_cpg_modes():
    import numpy as np

    from src.environment import CPG_CLOCK
    rng = np.random.default_rng(0)
    perm = list(np.random.default_rng(0).permutation(30))
    states = {m: CreatureEnv(cpg_mode=m, cpg_rng=np.random.default_rng(0), cpg_perm=perm).reset()
              for m in ("cpg", "zeros", "noise", "shuffled")}
    assert all(s.shape == (15,) for s in states.values())
    assert tuple(states["cpg"][13:15]) == tuple(np.float32(CPG_CLOCK[0]))
    assert tuple(states["zeros"][13:15]) == (0.0, 0.0)
    assert np.allclose(states["noise"][13:15], rng.uniform(-1, 1, 2))
    assert tuple(states["shuffled"][13:15]) == tuple(np.float32(CPG_CLOCK[perm[0]]))
    with pytest.raises(ValueError):
        CreatureEnv(cpg_mode="shuffled", cpg_perm=[0] * 30)  # not a permutation
    # every mode leaves the physics part of the state alone
    for s in states.values():
        assert (s[:13] == states["cpg"][:13]).all()


def test_cpg_default_unchanged():
    import numpy as np

    from src.environment import CPG_CLOCK
    env = CreatureEnv()
    env.reset()
    for i in range(1, 40):
        state, *_ = env.step([0.1, -0.1, 0.1, -0.1])
        assert (state[13:15] == np.array(CPG_CLOCK[i % 30], dtype=np.float32)).all()


def test_ground_termination_on_body_touch():
    env = CreatureEnv(fitness_mode="ground_termination")
    env.reset()
    # Force torso down to touch the ground (ground surface is at ~495-500)
    env.torso.position = (400, 496)
    action = [0.0, 0.0, 0.0, 0.0]
    state, reward, done, fallen = env.step(action)
    assert done is True
    assert fallen is True
    assert reward < -50.0  # Includes ground termination penalty

