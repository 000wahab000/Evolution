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
        assert len(state) == 13
        assert env.torso.position.x == 400
        assert env.torso.position.y == 440
        assert env.torso.angle == 0.0
        assert env.steps == 0


def test_environment_fitness_modes():
    for mode in ["baseline", "tilt_penalized", "energy_penalized", "all_penalties"]:
        env = CreatureEnv(fitness_mode=mode)
        env.reset()
        action = [0.5, -0.5, 0.2, -0.2]
        state, reward, done, fallen = env.step(action)
        assert isinstance(reward, float)

