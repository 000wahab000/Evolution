import os

from scripts.verify_results import verify
from src.research.config import ExperimentConfig
from src.research.runlog import read_json
from src.research.runner import FILES, run_dir, run_experiment

# Tiny on purpose: 3 generations of 4 genomes x 2 rollouts, 60-step episodes.
TINY = {
    "name": "tiny", "seeds": [0, 1], "optimizer": "ga", "optimizer_params": {"pop_size": 4},
    "episodes_budget": 24, "rollouts_k": 2, "max_steps": 60, "eval_every_episodes": 8,
    "test_terrain_seeds": [1000, 1001], "control_terrain_seeds": [2000],
    "validation_terrain_seeds": [3000],
    "variants": {"cpg": {}, "zeros": {"brain_input": "zeros"}},
    "hypotheses": [{"id": "H1", "a": "cpg", "b": "zeros", "metric": "test_distance",
                    "direction": "greater", "text": "cpg > zeros"}],
}


def tiny_results(out_root):
    exp = ExperimentConfig.from_dict(TINY)
    first = run_experiment(exp, str(out_root), log=lambda *_: None)
    return exp, first


def test_runs_end_to_end_and_resumes(tmp_path):
    exp, first = tiny_results(tmp_path)
    assert set(first.values()) == {"done"} and len(first) == 4
    d = run_dir(str(tmp_path), exp.resolve("cpg", 0))
    assert all(os.path.exists(os.path.join(d, f)) for f in FILES + ["DONE"])
    final = read_json(os.path.join(d, "final_test.json"))
    assert final["episodes_used"] == 24 and final["generations"] == 3
    assert {"test", "control", "train", "test_dt2"} <= set(final)
    with open(os.path.join(d, "evals.csv")) as f:
        assert len(f.read().strip().splitlines()) == 1 + 3  # header + evals at 8, 16, 24
    assert verify(os.path.join(str(tmp_path), "tiny")) == []

    again = run_experiment(exp, str(tmp_path), log=lambda *_: None)
    assert set(again.values()) == {"skipped"}


def test_crashed_run_is_redone(tmp_path):
    exp, _ = tiny_results(tmp_path)
    d = run_dir(str(tmp_path), exp.resolve("zeros", 1))
    os.remove(os.path.join(d, "DONE"))
    assert verify(os.path.join(str(tmp_path), "tiny")) != []
    again = run_experiment(exp, str(tmp_path), log=lambda *_: None)
    assert again["tiny/zeros/1"] == "done" and again["tiny/cpg/0"] == "skipped"
