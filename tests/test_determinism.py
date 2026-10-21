import os

from src.research.config import ExperimentConfig
from src.research.runner import run_dir, run_one
from tests.test_runner_smoke import TINY


def read(path):
    with open(path) as f:
        return f.read()


def test_same_seed_twice_gives_identical_history(tmp_path):
    cfg = ExperimentConfig.from_dict(TINY).resolve("cpg", 0)
    a, b = str(tmp_path / "a"), str(tmp_path / "b")
    run_one(cfg, a)
    run_one(cfg, b)
    for name in ("history.csv", "evals.csv", "final_test.json"):
        assert read(os.path.join(run_dir(a, cfg), name)) == read(os.path.join(run_dir(b, cfg), name))


def test_noise_variant_is_deterministic_too(tmp_path):
    cfg = ExperimentConfig.from_dict({**TINY, "variants": {"noise": {"brain_input": "noise"}}}).resolve("noise", 0)
    a, b = str(tmp_path / "a"), str(tmp_path / "b")
    run_one(cfg, a)
    run_one(cfg, b)
    assert read(os.path.join(run_dir(a, cfg), "history.csv")) == read(os.path.join(run_dir(b, cfg), "history.csv"))
