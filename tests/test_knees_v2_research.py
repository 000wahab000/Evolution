import os
from dataclasses import asdict

import numpy as np

from src.research.config import ExperimentConfig, RunConfig, config_sha256
from src.research.runner import run_one
from tests.test_runner_smoke import TINY


def test_default_hash_ignores_new_fields():
    cfg = RunConfig("e", "v", 0)
    d = asdict(cfg)
    del d["knees"], d["fall_rule"]
    assert cfg.sha256() == config_sha256(d)


def test_new_fields_change_hash():
    base = RunConfig("e", "v", 0).sha256()
    assert RunConfig("e", "v", 0, fall_rule="penalty").sha256() != base
    assert RunConfig("e", "v", 0, knees="spot").sha256() != base


def test_tiny_spot_penalty_run_deterministic(tmp_path):
    exp = ExperimentConfig.from_dict({**TINY, "variants": {"s": {"knees": "spot", "fall_rule": "penalty"}}})
    hists = []
    for n in ("a", "b"):
        root = str(tmp_path / n)
        assert run_one(exp.resolve("s", 0), root) == "done"
        d = os.path.join(root, "tiny", "s", "seed_0")
        with open(os.path.join(d, "history.csv")) as f:
            hists.append(f.read())
        assert np.load(os.path.join(d, "champion.npz"))["w1"].shape == (16, 23)
    assert hists[0] == hists[1]
