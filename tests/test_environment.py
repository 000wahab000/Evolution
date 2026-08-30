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

