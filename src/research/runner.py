"""Run one (variant, seed) into its own folder; run an experiment's runs in parallel processes.

Each run writes only results/<exp>/<variant>/seed_<n>/, and DONE goes last, so parallel runs
never share a file and a crashed run (no DONE) is simply redone. Only the parent writes
manifest.json.
"""

import os
import shutil
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

from src.research.config import SCHEMA_VERSION, make_rng
from src.research.evaluate import (
    DIM,
    EpisodeCounter,
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
    replace_with_retry,
    utc_now,
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


def run_dir(out_root, cfg):
    return os.path.join(out_root, cfg.experiment, cfg.variant, f"seed_{cfg.seed}")


def _csv(cols, rows):
    return "\n".join([",".join(cols)] + [",".join(repr(v) for v in r) for r in rows]) + "\n"


def run_one(cfg, out_root="results"):
    """Train one run to its episode budget and write its folder. Returns 'done' or 'skipped'."""
    d = run_dir(out_root, cfg)
    if is_done(d):
        return "skipped"
    if os.path.exists(d):
        shutil.rmtree(d)  # leftovers of a crashed run
    os.makedirs(d)
    write_json(os.path.join(d, "config.json"), cfg.to_dict())
    started, t0 = utc_now(), time.time()

    opt = make_optimizer(cfg.optimizer, DIM, cfg.optimizer_params, make_rng(cfg.seed, "optimizer"))
    train_rng = make_rng(cfg.seed, "train")
    val, test, control = (fixed_scenarios(cfg, s) for s in ("validation", "test", "control"))
    counter, eval_eps = EpisodeCounter(), 0
    history, evals, gen, next_eval = [], [], 0, cfg.eval_every_episodes
    champ, champ_val, diversity = None, -np.inf, float("nan")

    def held_out(genome, scenarios, substeps=1):
        nonlocal eval_eps
        eval_eps += len(scenarios)
        return summarize(evaluate_genome(genome, cfg, scenarios, substeps=substeps)[1])

    while True:
        genomes = opt.ask()
        if counter.n + len(genomes) * cfg.rollouts_k > cfg.episodes_budget:
            break
        scenarios = train_scenarios(cfg, train_rng)  # shared by the whole generation
        fits = [evaluate_genome(g, cfg, scenarios, counter)[0] for g in genomes]
        opt.tell(fits)
        gen += 1
        # Champion = this generation's train-best, kept only if it also wins on validation
        # terrains (train distribution). Test and control never influence selection.
        top = genomes[int(np.argmax(fits))]
        v = evaluate_genome(top, cfg, val)[0]
        eval_eps += len(val)
        if v > champ_val:
            champ, champ_val = np.array(top, dtype=float), v
        diversity = float(np.std(np.stack(genomes), axis=0, ddof=1).mean())
        history.append([counter.n, gen, float(max(fits)), float(np.mean(fits)), diversity])
        if counter.n >= next_eval:
            t, c = held_out(champ, test), held_out(champ, control)
            evals.append([counter.n, t["distance"], t["fall_rate"], t["cot"],
                          c["distance"], c["fall_rate"], c["cot"]])
            next_eval = (counter.n // cfg.eval_every_episodes + 1) * cfg.eval_every_episodes
    if champ is None:
        raise ValueError(f"{cfg.run_id}: episodes_budget too small for one generation")

    final = {
        "schema_version": SCHEMA_VERSION,
        "test": held_out(champ, test),
        "control": held_out(champ, control),
        "train": held_out(champ, val),  # fresh train-distribution terrains, not the noisy argmax
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


def _write_manifest(exp, out_root):
    runs = {f"{v}/seed_{s}": "done" if is_done(run_dir(out_root, exp.resolve(v, s))) else "pending"
            for v in exp.variants for s in exp.seeds}
    write_json(os.path.join(out_root, exp.name, "manifest.json"), {
        "schema_version": SCHEMA_VERSION, "experiment": exp.name,
        "config_sha256": exp.sha256(), **git_info(), "runs": runs})


def run_experiment(exp, out_root="results", workers=1, only=None, seeds=None, log=print):
    exp_dir = os.path.join(out_root, exp.name)
    os.makedirs(exp_dir, exist_ok=True)
    write_json(os.path.join(exp_dir, "experiment.json"), exp.to_dict())
    cfgs = [exp.resolve(v, s) for v in exp.variants if not only or v in only
            for s in (exp.seeds if seeds is None else seeds)]
    _write_manifest(exp, out_root)
    results = {}
    if workers <= 1:
        for cfg in cfgs:
            results[cfg.run_id] = run_one(cfg, out_root)
            log(f"{cfg.run_id}: {results[cfg.run_id]}")
            _write_manifest(exp, out_root)
    else:
        with ProcessPoolExecutor(workers) as pool:
            futures = {pool.submit(run_one, cfg, out_root): cfg for cfg in cfgs}
            for fut in as_completed(futures):
                cfg = futures[fut]
                results[cfg.run_id] = fut.result()
                log(f"{cfg.run_id}: {results[cfg.run_id]}")
                _write_manifest(exp, out_root)
    return results
