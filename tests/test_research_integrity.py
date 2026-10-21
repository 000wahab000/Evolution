"""Guards for the things that would silently corrupt an experiment's conclusions."""

import json
import os

import numpy as np
import pytest

from scripts.analyze import main as analyze
from scripts.record_replay import main as record_replay
from src.research.analysis import MISSING, load, pairwise, verdict
from src.research.config import ExperimentConfig
from src.research.evaluate import fixed_scenarios, make_env, rollout
from src.research.runlog import read_json
from src.research.runner import StaleRunError, run_dir, run_experiment, run_one, verify_experiment
from src.utils import load_state_dict_file
from tests.test_runner_smoke import TINY

QUIET = {"log": lambda *_: None}


def read(path):
    with open(path, encoding="utf-8") as f:
        return f.read()


# ── verdict rule ─────────────────────────────────────────────────────────────

def row(direction, p_holm, lo, hi, missing=False):
    return {"direction": direction, "p_holm": p_holm, "diff_ci_lo": lo, "diff_ci_hi": hi, "missing": missing}


def test_verdict_rule():
    assert verdict(row("greater", 0.01, 5, 20)) == "supported"
    assert verdict(row("greater", 0.01, -20, -5)) == "not supported"
    assert verdict(row("greater", 0.01, -5, 20)) == "inconclusive"  # CI straddles 0
    assert verdict(row("greater", 0.20, 5, 20)) == "inconclusive"  # not significant
    assert verdict(row("less", 0.01, -20, -5)) == "supported"
    assert verdict(row("less", 0.01, 5, 20)) == "not supported"
    assert verdict(row("greater", 0.001, 5, 20, missing=True)) == MISSING


# ── champion selection never looks at test or control ───────────────────────

def test_changing_held_out_terrains_leaves_the_champion_unchanged(tmp_path):
    base = ExperimentConfig.from_dict(TINY).resolve("cpg", 0)
    other = ExperimentConfig.from_dict({**TINY, "test_terrain_seeds": [1500, 1501],
                                        "control_terrain_seeds": [2500]}).resolve("cpg", 0)
    a, b = str(tmp_path / "a"), str(tmp_path / "b")
    run_one(base, a)
    run_one(other, b)
    ca = load_state_dict_file(os.path.join(run_dir(a, base), "champion.npz"))
    cb = load_state_dict_file(os.path.join(run_dir(b, other), "champion.npz"))
    assert all((ca[k] == cb[k]).all() for k in ca)
    assert read(os.path.join(run_dir(a, base), "history.csv")) == read(os.path.join(run_dir(b, other), "history.csv"))


# ── equal budgets and equal champion picks for every optimizer ───────────────

SMALL = {"ga": {"pop_size": 4}, "cmaes": {"popsize": 4}, "openai_es": {"pop_size": 4},
         "random_search": {"n_directions": 2}}


@pytest.mark.parametrize("name", sorted(SMALL))
def test_budget_and_picks_per_optimizer(tmp_path, name):
    cfg = ExperimentConfig.from_dict({**TINY, "variants": {name: {"optimizer": name,
                                                                  "optimizer_params": SMALL[name]}}}).resolve(name, 0)
    run_one(cfg, str(tmp_path))
    d = run_dir(str(tmp_path), cfg)
    final = read_json(os.path.join(d, "final_test.json"))
    assert final["episodes_used"] == 24 and final["generations"] == 3  # 4 genomes x k=2 x 3 gens
    # 3 champion checks on the grid (8, 16, 24) x 1 validation terrain, + 3 x (2 test + 1 control)
    # evals, + final 2 test + 1 control + 1 train_eval + 2 dt/2 test.
    assert final["eval_episodes"] == 3 * 1 + 3 * 3 + 6


# ── resume, stale configs and failing runs ───────────────────────────────────

def test_stale_finished_run_is_refused_and_flagged(tmp_path):
    out = str(tmp_path)
    run_experiment(ExperimentConfig.from_dict(TINY), out, **QUIET)
    edited = ExperimentConfig.from_dict({**TINY, "max_steps": 61})
    with pytest.raises(StaleRunError):
        run_one(edited.resolve("cpg", 0), out)
    results = run_experiment(edited, out, **QUIET)
    assert all(r.startswith("failed: StaleRunError") for r in results.values())
    assert any("stale run" in p for p in verify_experiment(os.path.join(out, "tiny")))
    manifest = read_json(os.path.join(out, "tiny", "manifest.json"))
    assert set(manifest["runs"].values()) == {"failed"}


def test_one_failing_run_does_not_stop_the_others(tmp_path):
    exp = ExperimentConfig.from_dict({**TINY, "variants": {
        "cpg": {}, "broken": {"optimizer": "no_such_optimizer"}}})
    results = run_experiment(exp, str(tmp_path), **QUIET)
    assert results["tiny/cpg/0"] == "done" and results["tiny/cpg/1"] == "done"
    assert results["tiny/broken/0"].startswith("failed: ValueError")
    manifest = read_json(os.path.join(str(tmp_path), "tiny", "manifest.json"))
    assert manifest["runs"]["cpg/seed_0"] == "done" and manifest["runs"]["broken/seed_1"] == "failed"


def test_parallel_workers_are_deterministic(tmp_path):
    exp = ExperimentConfig.from_dict({**TINY, "seeds": [0]})
    a, b = str(tmp_path / "a"), str(tmp_path / "b")
    run_experiment(exp, a, workers=2, **QUIET)
    run_experiment(exp, b, workers=1, **QUIET)
    for v in exp.variants:
        cfg = exp.resolve(v, 0)
        assert read(os.path.join(run_dir(a, cfg), "history.csv")) == read(os.path.join(run_dir(b, cfg), "history.csv"))


# ── missing data never shrinks the Holm family silently ──────────────────────

def test_missing_hypothesis_stays_in_holm_family(tmp_path):
    hyps = TINY["hypotheses"] + [{"id": "H9", "a": "cpg", "b": "ghost", "metric": "test_distance",
                                  "direction": "greater", "text": "no data"}]
    exp = ExperimentConfig.from_dict({**TINY, "seeds": [0, 1, 2], "hypotheses": hyps})
    run_experiment(exp, str(tmp_path), **QUIET)
    exp, runs = load(os.path.join(str(tmp_path), "tiny"))
    _, confirm = pairwise(exp, runs, np.random.default_rng(0))
    assert [r["id"] for r in confirm] == ["H1", "H9"]
    h9 = confirm[1]
    assert h9["verdict"] == MISSING and h9["p"] == 1.0 and h9["p_holm"] == 1.0
    assert confirm[0]["p_holm"] == pytest.approx(min(1.0, 2 * confirm[0]["p"]))  # family of 2, not 1


def test_analyze_refuses_incomplete_experiment_unless_allowed(tmp_path):
    out = str(tmp_path)
    run_experiment(ExperimentConfig.from_dict({**TINY, "seeds": [0, 1, 2]}), out, **QUIET)
    exp_dir = os.path.join(out, "tiny")
    os.remove(os.path.join(run_dir(out, ExperimentConfig.from_dict(TINY).resolve("zeros", 2)), "DONE"))
    with pytest.raises(SystemExit):
        analyze([exp_dir, "--no-replay"])
    analyze([exp_dir, "--no-replay", "--allow-incomplete"])
    text = read(os.path.join(exp_dir, "summary", "REPORT.md"))
    assert "Incomplete data" in text and "zeros/seed_2: missing DONE" in text


# ── shuffled control, replay fidelity ────────────────────────────────────────

def test_shuffled_permutation_is_fixed_per_run_not_per_episode():
    exp = ExperimentConfig.from_dict({**TINY, "variants": {"shuffled": {"brain_input": "shuffled"}}})
    cfg0, cfg1 = exp.resolve("shuffled", 0), exp.resolve("shuffled", 1)
    s_a, s_b = fixed_scenarios(cfg0, "test")
    assert make_env(cfg0, s_a).cpg_perm == make_env(cfg0, s_b).cpg_perm
    assert make_env(cfg0, s_a).cpg_perm != make_env(cfg1, s_a).cpg_perm


def test_record_replay_champion_matches_the_measured_rollout(tmp_path):
    exp = ExperimentConfig.from_dict({**TINY, "variants": {"noise": {"brain_input": "noise"}}})
    cfg = exp.resolve("noise", 0)
    run_one(cfg, str(tmp_path))
    champ = os.path.join(run_dir(str(tmp_path), cfg), "champion.npz")
    out = str(tmp_path / "replay.json")
    record_replay(["--champion", champ, "--out", out])
    replay = json.loads(read(out))
    measured = rollout(load_state_dict_file(champ), cfg, fixed_scenarios(cfg, "test")[0])
    assert replay["steps"] == measured["steps"]
    assert replay["distance"] == round(measured["distance"], 2)
    assert "tiny / noise / seed 0" in replay["label"]
