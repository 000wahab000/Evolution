import json
import os

import pytest

from src.research.config import ExperimentConfig, make_rng
from src.research.runlog import read_json, verify_done, write_done, write_json

CFG = {
    "name": "t",
    "seeds": [0, 1],
    "episodes_budget": 100,
    "variants": {"cpg": {}, "zeros": {"brain_input": "zeros"}},
}


def test_json_round_trip():
    cfg = ExperimentConfig.from_dict(CFG)
    again = ExperimentConfig.from_dict(json.loads(json.dumps(cfg.to_dict())))
    assert again == cfg
    run = again.resolve("zeros", 1)
    assert (run.brain_input, run.episodes_budget, run.seed) == ("zeros", 100, 1)
    assert run.sha256() == cfg.resolve("zeros", 1).sha256()


def test_unknown_keys_and_versions_fail_loudly():
    with pytest.raises(ValueError):
        ExperimentConfig.from_dict({**CFG, "variants": {"x": {"bogus": 1}}})
    with pytest.raises(ValueError):
        ExperimentConfig.from_dict({**CFG, "schema_version": 99})


def test_same_seed_same_stream_different_keys_differ():
    a = make_rng(3, "train", 7).random(5)
    assert (a == make_rng(3, "train", 7).random(5)).all()
    assert not (a == make_rng(3, "test", 7).random(5)).all()
    assert not (a == make_rng(4, "train", 7).random(5)).all()


def test_done_checksums(tmp_path):
    d = str(tmp_path)
    write_json(os.path.join(d, "a.json"), {"x": 1})
    write_done(d, ["a.json"])
    assert verify_done(d) == []
    assert read_json(os.path.join(d, "a.json")) == {"x": 1}
    with open(os.path.join(d, "a.json"), "w") as f:
        f.write("{}")
    assert verify_done(d) == ["checksum mismatch a.json"]
