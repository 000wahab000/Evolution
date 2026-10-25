import os
from dataclasses import asdict

import numpy as np
import pytest

from src.research.config import ExperimentConfig, config_sha256
from src.research.evaluate import KNEE_DIM, from_flat, to_flat
from src.research.runner import run_dir, run_one
from tests.test_runner_smoke import TINY

KNEE_TINY = {**TINY, "variants": {"knees": {"knees": True}}}


def test_straight_hash_unchanged():
    cfg = ExperimentConfig.from_dict(TINY).resolve("cpg", 0)
    d = asdict(cfg)
    del d["knees"], d["fall_rule"]
    assert cfg.sha256() == config_sha256(d)


def test_knees_hash_differs():
    straight = ExperimentConfig.from_dict(TINY).resolve("cpg", 0)
    kneed = ExperimentConfig.from_dict(KNEE_TINY).resolve("knees", 0)
    assert kneed.knees and kneed.sha256() != straight.sha256()


def test_flat_round_trip_knees():
    vec = np.random.default_rng(0).normal(size=KNEE_DIM)
    sd = from_flat(vec)
    assert [sd[k].shape for k in ("w1", "b1", "w2", "b2")] == [(16, 23), (16,), (8, 16), (8,)]
    assert np.allclose(to_flat(sd), vec, atol=1e-6)


def test_run_one_knees_deterministic(tmp_path):
    cfg = ExperimentConfig.from_dict(KNEE_TINY).resolve("knees", 0)
    a, b = str(tmp_path / "a"), str(tmp_path / "b")
    run_one(cfg, a)
    run_one(cfg, b)
    with np.load(os.path.join(run_dir(a, cfg), "champion.npz")) as z:
        assert z["w1"].shape == (16, 23)
    hist = lambda d: open(os.path.join(run_dir(d, cfg), "history.csv")).read()  # noqa: E731
    assert hist(a) == hist(b)


def test_unknown_variant_key_rejected():
    with pytest.raises(ValueError):
        ExperimentConfig.from_dict({**TINY, "variants": {"k": {"knees": True, "bogus": 1}}})
