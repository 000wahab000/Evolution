"""Run one (variant, seed) into its own folder; run an experiment's runs in parallel processes.

Each run writes only results/<exp>/<variant>/seed_<n>/, and DONE goes last, so parallel runs
never share a file and a crashed run (no DONE) is simply redone. A finished run whose config
hash differs from the current config is refused, never silently kept. Only the parent writes
manifest.json.
"""

import os
import shutil
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

from src.research.config import SCHEMA_VERSION, ExperimentConfig, make_rng
from src.research.evaluate import (
    EpisodeCounter,
    dim,
    evaluate_genome,
    fixed_scenarios,
    from_flat,
    summarize,
    train_scenarios,
)
from src.research.optimizers import make_optimizer
from src.research.runlog import (
    git_info,
    is_done,
    read_json,
    replace_with_retry,
    utc_now,
    verify_done,
    write_done,
    write_json,
    write_meta,
    write_text,
)
from src.terrain import TERRAIN_VERSION

HISTORY_COLS = ["episodes", "generation", "best_train", "mean_train", "diversity"]
EVAL_COLS = ["episodes", "test_distance", "test_fall_rate", "test_cot",
             "control_distance", "control_fall_rate", "control_cot"]
FILES = ["config.json", "meta.json", "history.csv", "evals.csv", "champion.npz", "final_test.json"]


class StaleRunError(RuntimeError):
    """A finished run folder was produced by a different config than the one being run."""


def run_dir(out_root, cfg):
    return os.path.join(out_root, cfg.experiment, cfg.variant, f"seed_{cfg.seed}")


def stale_reason(d, cfg):
    """None when the finished run in d matches cfg, else why it does not."""
    have = read_json(os.path.join(d, "meta.json")).get("config_sha256")
    return None if have == cfg.sha256() else f"config_sha256 {str(have)[:12]} != current {cfg.sha256()[:12]}"


def _csv(cols, rows):
    return "\n".join([",".join(cols)] + [",".join(repr(v) for v in r) for r in rows]) + "\n"


def run_one(cfg, out_root="results"):
    """Train one run to its episode budget and write its folder. Returns 'done' or 'skipped'."""
    d = run_dir(out_root, cfg)
    if is_done(d):
        reason = stale_reason(d, cfg)
        if reason:
            raise StaleRunError(f"{d} holds a finished run from another config ({reason}); "
                                "delete that folder to rerun it")
        return "skipped"
    if os.path.exists(d):
        shutil.rmtree(d)  # leftovers of a crashed run
    os.makedirs(d)
    write_json(os.path.join(d, "config.json"), cfg.to_dict())
    started, t0 = utc_now(), time.time()

    opt = make_optimizer(cfg.optimizer, dim(cfg), cfg.optimizer_params, make_rng(cfg.seed, "optimizer"))
    train_rng = make_rng(cfg.seed, "train")
    val, test, control, train_eval = (fixed_scenarios(cfg, s)
                                      for s in ("validation", "test", "control", "train_eval"))
    counter, eval_eps = EpisodeCounter(), 0
    history, evals, gen, next_eval = [], [], 0, cfg.eval_every_episodes
    champ, champ_val, diversity, checked_at = None, -np.inf, float("nan"), -1

    def held_out(genome, scenarios, substeps=1):
        nonlocal eval_eps
        eval_eps += len(scenarios)
        return summarize(evaluate_genome(genome, cfg, scenarios, substeps=substeps)[1])

    def check_champion():
        # Champion = the optimizer's recommendation (GA best / ES mean), kept only if it wins on
        # validation terrains (train distribution). Checked on the eval grid only, so every
        # optimizer gets the same number of picks. Test and control never influence selection.
        nonlocal champ, champ_val, eval_eps, checked_at
        cand = opt.recommend()
        v = evaluate_genome(cand, cfg, val)[0]
        eval_eps += len(val)
        if v > champ_val:
            champ, champ_val = np.array(cand, dtype=float), v
        checked_at = gen

    while True:
        genomes = opt.ask()
        if counter.n + len(genomes) * cfg.rollouts_k > cfg.episodes_budget:
            break
        scenarios = train_scenarios(cfg, train_rng)  # shared by the whole generation
        fits = [evaluate_genome(g, cfg, scenarios, counter)[0] for g in genomes]
        opt.tell(fits)
        gen += 1
        diversity = float(np.std(np.stack(genomes), axis=0, ddof=1).mean())
        history.append([counter.n, gen, float(max(fits)), float(np.mean(fits)), diversity])
        if counter.n >= next_eval:
            check_champion()
            t, c = held_out(champ, test), held_out(champ, control)
            evals.append([counter.n, t["distance"], t["fall_rate"], t["cot"],
                          c["distance"], c["fall_rate"], c["cot"]])
            next_eval = (counter.n // cfg.eval_every_episodes + 1) * cfg.eval_every_episodes
    if gen == 0:
        raise ValueError(f"{cfg.run_id}: episodes_budget too small for one generation")
    if checked_at != gen:  # the budget ended between grid points: one last pick
        check_champion()

    final = {
        "schema_version": SCHEMA_VERSION,
        "test": held_out(champ, test),
        "control": held_out(champ, control),
        "train": held_out(champ, train_eval),  # fresh train-distribution terrains, never used to pick
        "test_dt2": held_out(champ, test, substeps=2),
        "diversity": diversity,
        "generations": gen,
        "episodes_used": counter.n,
        "eval_episodes": eval_eps,
        "champion_validation_return": float(champ_val),
    }
    write_text(os.path.join(d, "history.csv"), _csv(HISTORY_COLS, history))
    write_text(os.path.join(d, "evals.csv"), _csv(EVAL_COLS, evals))
    tmp = os.path.join(d, "champion.tmp.npz")
    np.savez(tmp, **from_flat(champ))
    replace_with_retry(tmp, os.path.join(d, "champion.npz"))
    write_json(os.path.join(d, "final_test.json"), final)
    write_meta(d, cfg, started, time.time() - t0, counter.n, eval_eps, TERRAIN_VERSION)
    write_done(d, FILES)
    return "done"


def _write_manifest(exp, out_root, failures):
    def status(v, s):
        key = f"{v}/seed_{s}"
        if key in failures:
            return "failed"
        return "done" if is_done(run_dir(out_root, exp.resolve(v, s))) else "pending"
    write_json(os.path.join(out_root, exp.name, "manifest.json"), {
        "schema_version": SCHEMA_VERSION, "experiment": exp.name,
        "config_sha256": exp.sha256(), **git_info(),
        "runs": {f"{v}/seed_{s}": status(v, s) for v in exp.variants for s in exp.seeds},
        "failures": failures})


def run_experiment(exp, out_root="results", workers=1, only=None, seeds=None, log=print):
    """Run the selected runs. A failing run is logged and recorded; the others carry on.
    Returns {run_id: 'done' | 'skipped' | 'failed: <error>'}."""
    exp_dir = os.path.join(out_root, exp.name)
    os.makedirs(exp_dir, exist_ok=True)
    write_json(os.path.join(exp_dir, "experiment.json"), exp.to_dict())
    cfgs = [exp.resolve(v, s) for v in exp.variants if not only or v in only
            for s in (exp.seeds if seeds is None else seeds)]
    results, failures = {}, {}
    _write_manifest(exp, out_root, failures)

    def record(cfg, fn):
        try:
            results[cfg.run_id] = fn()
        except Exception as e:  # noqa: BLE001 - one bad run must not stop the night's batch
            results[cfg.run_id] = f"failed: {type(e).__name__}: {e}"
            failures[f"{cfg.variant}/seed_{cfg.seed}"] = results[cfg.run_id]
        log(f"{cfg.run_id}: {results[cfg.run_id]}")
        _write_manifest(exp, out_root, failures)

    if workers <= 1:
        for cfg in cfgs:
            record(cfg, lambda cfg=cfg: run_one(cfg, out_root))
    else:
        with ProcessPoolExecutor(workers) as pool:
            futures = {pool.submit(run_one, cfg, out_root): cfg for cfg in cfgs}
            for fut in as_completed(futures):
                record(futures[fut], fut.result)
    return results


def verify_experiment(exp_dir):
    """Problems that make an experiment unfit to analyze or publish (empty list when complete)."""
    exp = ExperimentConfig.from_dict(read_json(os.path.join(exp_dir, "experiment.json")))
    out_root = os.path.dirname(os.path.abspath(exp_dir))
    problems = []
    for v in exp.variants:
        for s in exp.seeds:
            cfg = exp.resolve(v, s)
            d = run_dir(out_root, cfg)
            issues = verify_done(d)
            if not issues:
                reason = stale_reason(d, cfg)
                issues = [f"stale run: {reason}"] if reason else []
            problems += [f"{v}/seed_{s}: {p}" for p in issues]
    manifest_path = os.path.join(exp_dir, "manifest.json")
    if not os.path.exists(manifest_path):
        return problems + ["missing manifest.json"]
    manifest = read_json(manifest_path)
    if manifest.get("config_sha256") != exp.sha256():
        problems.append("manifest.json config hash does not match experiment.json")
    problems += [f"manifest: {k} {status}" for k, status in manifest.get("runs", {}).items()
                 if status != "done"]
    return problems
