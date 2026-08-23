import pytest
from src.environment import CreatureEnv

def test_environment_initialization():
    env = CreatureEnv(fitness_mode="baseline")
    assert env.steps == 0
    assert env.fitness_mode == "baseline"

def test_environment_reset():
    env = CreatureEnv()
    state = env.reset()
    assert len(state) == 13

def test_environment_step():
    env = CreatureEnv()
    env.reset()
    # Dummy action: 4 legs, float values between -1 and 1
    action = [0.0, 0.0, 0.0, 0.0]
    state, reward, done, fallen = env.step(action)
    assert len(state) == 13
    assert isinstance(reward, float)
    assert isinstance(done, bool)
    assert isinstance(fallen, bool)
