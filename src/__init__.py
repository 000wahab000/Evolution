
try:  # registers Evolution1-Quadruped-v0 when gymnasium is installed
    from src import gym_env  # noqa: F401
except ImportError:
    pass
