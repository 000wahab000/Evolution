"""Read a results folder and compute per-seed metrics, summaries, pairwise stats and verdicts.

Everything here reads files only, except robustness() and gait(), which replay saved champions.
"""

import csv
import itertools
import os

import numpy as np

from src.research.config import ExperimentConfig, make_rng
from src.research.evaluate import Scenario, fixed_scenarios, rollout, summarize
from src.research.runlog import is_done, read_json
from src.research.runner import run_dir
from src.research.stats import (
    bootstrap_ci,
    bootstrap_diff_ci,
    cliffs_delta,
    holm,
    iqm,
    mann_whitney,
    prob_improvement,
)
from src.utils import load_state_dict_file

TAU = 300.0  # M5 threshold, px
FRICTIONS = [0.6, 0.8, 1.0, 1.2, 1.4]
POSE_SIGMAS = [0.0, 0.05, 0.1, 0.2]
EXPLOIT_DROP = 0.30  # flag a seed whose test distance falls more than this at dt/2
EXPLOIT_MIN_DISTANCE = 50.0  # px; relative drops of near-zero walkers are noise, not exploits
MISSING = "not tested (missing data)"
METRICS = ["test_distance", "train_distance", "control_distance", "shift_drop", "gap_test",
           "gap_control", "fall_rate", "cot", "dt2_distance", "diversity"]


def _read_csv(path):
    with open(path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return {k: np.array([float(r[k]) for r in rows]) for k in (rows[0] if rows else {})}


def load(exp_dir):
    """{variant: [run dict]} for every finished run (folders with DONE)."""
    exp = ExperimentConfig.from_dict(read_json(os.path.join(exp_dir, "experiment.json")))
    out_root = os.path.dirname(os.path.abspath(exp_dir))
    runs = {}
    for v in exp.variants:
        for s in exp.seeds:
            cfg = exp.resolve(v, s)
            d = run_dir(out_root, cfg)
            if not is_done(d):
                continue
            f = read_json(os.path.join(d, "final_test.json"))
            ev = _read_csv(os.path.join(d, "evals.csv"))
            reached = np.nonzero(ev.get("test_distance", np.array([])) >= TAU)[0]
            runs.setdefault(v, []).append({
                "seed": s, "cfg": cfg, "dir": d, "final": f,
                "history": _read_csv(os.path.join(d, "history.csv")), "evals": ev,
                "meta": read_json(os.path.join(d, "meta.json")),
                "test_distance": f["test"]["distance"], "train_distance": f["train"]["distance"],
                "control_distance": f["control"]["distance"],
                "shift_drop": f["control"]["distance"] - f["test"]["distance"],
                "gap_test": f["train"]["distance"] - f["test"]["distance"],
                "gap_control": f["train"]["distance"] - f["control"]["distance"],
                "fall_rate": f["test"]["fall_rate"], "falls": f["test"]["falls"],
                "episodes": f["test"]["episodes"], "cot": f["test"]["cot"],
                "dt2_distance": f["test_dt2"]["distance"], "diversity": f["diversity"],
                "episodes_to_tau": float(ev["episodes"][reached[0]]) if len(reached) else None,
            })
    return exp, runs


def values(runs, metric):
    x = np.array([r[metric] for r in runs], dtype=float)
    return x[~np.isnan(x)]


def summarize_experiment(exp, runs, rng):
    out = {}
    for v, rs in runs.items():
        s = {"n": len(rs)}
        for m in METRICS:
            x = values(rs, m)
            s[m] = ({"iqm": float(iqm(x)), "ci": bootstrap_ci(x, rng), "median": float(np.median(x))}
                    if len(x) else None)
        # Episodes are clustered by seed, so the CI resamples seeds, not episodes.
        per_seed = values(rs, "fall_rate")
        s["fall"] = {"mean": float(per_seed.mean()) if len(per_seed) else float("nan"),
                     "ci": bootstrap_ci(per_seed, rng, stat=_mean) if len(per_seed) else (float("nan"),) * 2,
                     "falls": sum(r["falls"] for r in rs), "episodes": sum(r["episodes"] for r in rs)}
        reached = [r["episodes_to_tau"] for r in rs if r["episodes_to_tau"] is not None]
        s["episodes_to_tau"] = {"tau": TAU, "reached": len(reached), "of": len(rs),
                                "median": float(np.median(reached)) if reached else None}
        t, d2 = s["test_distance"], s["dt2_distance"]
        s["dt2_drop"] = ((t["iqm"] - d2["iqm"]) / abs(t["iqm"])) if t and d2 and t["iqm"] else None
        eligible = [r for r in rs if r["test_distance"] > EXPLOIT_MIN_DISTANCE]
        flagged = [r["seed"] for r in eligible
                   if (r["test_distance"] - r["dt2_distance"]) / r["test_distance"] > EXPLOIT_DROP]
        s["dt2_flagged"] = {"seeds": flagged, "of": len(eligible)}
        s["exploit_flag"] = bool(flagged)
        out[v] = s
    return out


def _mean(x, axis=None):
    return np.mean(x, axis=axis)


def pairwise(exp, runs, rng):
    """Every ordered-once pair on test distance (exploratory, Holm over all pairs), plus each
    hypothesis contrast (confirmatory, Holm within the experiment's hypotheses)."""
    rows = []
    for a, b in itertools.combinations([v for v in exp.variants if v in runs], 2):
        rows.append(_contrast(runs, a, b, "test_distance", rng, kind="exploratory"))
    for row, p in zip(rows, holm([r["p"] for r in rows])):
        row["p_holm"] = p
    # Every pre-registered hypothesis stays in the Holm family; missing data gives p = 1.
    confirm = [_contrast(runs, h["a"], h["b"], h["metric"], rng, kind="confirmatory", hyp=h)
               for h in exp.hypotheses]
    for row, p in zip(confirm, holm([r["p"] for r in confirm])):
        row["p_holm"] = p
        row["verdict"] = verdict(row)
    return rows, confirm


def _contrast(runs, a, b, metric, rng, kind, hyp=None):
    xa, xb = values(runs.get(a, []), metric), values(runs.get(b, []), metric)
    head = {"kind": kind, "id": hyp["id"] if hyp else "", "a": a, "b": b, "metric": metric,
            "direction": hyp["direction"] if hyp else "", "n_a": len(xa), "n_b": len(xb)}
    if not len(xa) or not len(xb):
        nan = float("nan")
        return {**head, "missing": True, "iqm_a": nan, "iqm_b": nan, "iqm_diff": nan, "diff_ci_lo": nan,
                "diff_ci_hi": nan, "p": 1.0, "cliffs_delta": nan, "p_improvement": nan}
    lo, hi = bootstrap_diff_ci(xa, xb, rng)
    return {**head, "missing": False,
            "iqm_a": float(iqm(xa)), "iqm_b": float(iqm(xb)), "iqm_diff": float(iqm(xa) - iqm(xb)),
            "diff_ci_lo": lo, "diff_ci_hi": hi, "p": mann_whitney(xa, xb),
            "cliffs_delta": cliffs_delta(xa, xb), "p_improvement": prob_improvement(xa, xb)}


def verdict(row, alpha=0.05):
    """supported: Holm p < alpha and the IQM-difference CI excludes 0 in the predicted direction."""
    if row.get("missing"):
        return MISSING
    if row["p_holm"] >= alpha:
        return "inconclusive"
    up = row["diff_ci_lo"] > 0
    down = row["diff_ci_hi"] < 0
    if (row["direction"] == "greater" and up) or (row["direction"] == "less" and down):
        return "supported"
    if (row["direction"] == "greater" and down) or (row["direction"] == "less" and up):
        return "not supported"
    return "inconclusive"


def learning_curves(exp, runs, rng, n_boot=2000):
    """Test distance of the train-selected champion on a shared episode grid: IQM and 95% CI."""
    every = exp.resolve(next(iter(runs)), exp.seeds[0]).eval_every_episodes
    top = max(r["evals"]["episodes"].max() for rs in runs.values() for r in rs if len(r["evals"]))
    grid = np.arange(every, top + every, every)
    out = {"episodes": grid.tolist()}
    for v, rs in runs.items():
        curves = np.array([np.interp(grid, r["evals"]["episodes"], r["evals"]["test_distance"])
                           for r in rs if len(r["evals"])])
        cis = [bootstrap_ci(curves[:, i], rng, n=n_boot) for i in range(len(grid))]
        out[v] = {"iqm": iqm(curves, axis=0).tolist(), "lo": [c[0] for c in cis],
                  "hi": [c[1] for c in cis]}
    return out


def _champion(run):
    return load_state_dict_file(os.path.join(run["dir"], "champion.npz"))


def robustness(runs):
    """M6: test distance per run under friction scale and initial-pose noise (dt/2 comes from
    final_test.json). Replays champions on the test terrains."""
    out = {}
    for v, rs in runs.items():
        fr, pose = [], []
        for r in rs:
            cfg, params = r["cfg"], _champion(r)
            test = fixed_scenarios(cfg, "test")
            fr.append([r["test_distance"] if f == 1.0 else summarize(
                [rollout(params, cfg, Scenario(s.terrain, f, None, s.cpg_seed)) for s in test])["distance"]
                for f in FRICTIONS])
            pose.append([r["test_distance"] if sig == 0.0 else summarize(
                [rollout(params, cfg, Scenario(s.terrain, 1.0, tuple(make_rng(r["seed"], "pose", i, j)
                                                                     .normal(0, sig, 4)), s.cpg_seed))
                 for j, s in enumerate(test)])["distance"]
                for i, sig in enumerate(POSE_SIGMAS)])
        out[v] = {"friction": np.array(fr), "pose": np.array(pose),
                  "dt": np.array([[r["test_distance"], r["dt2_distance"]] for r in rs])}
    return out


def gait_signature(leg_angles):
    """M7 for one episode: dominant frequency per leg (cycles/step), leg0-leg2 phase lag and
    leg0 lag to the CPG clock (radians), or None when the episode is too short."""
    if len(leg_angles) < 60:
        return None
    x = leg_angles - leg_angles.mean(axis=0)
    spec = np.fft.rfft(x, axis=0)
    freqs = np.fft.rfftfreq(len(x))
    dom = np.argmax(np.abs(spec[1:]), axis=0) + 1
    k0 = dom[0]
    t = np.arange(1, len(x) + 1)
    clock = np.fft.rfft(np.sin(2 * np.pi * t / 30.0))
    kc = int(round(len(x) / 30.0))
    return {"freq": freqs[dom].tolist(), "ratio_to_cpg": float(freqs[k0] * 30.0),
            "phase_lag_02": float(np.angle(spec[k0, 0] * np.conj(spec[k0, 2]))),
            "lag_to_cpg": float(np.angle(spec[kc, 0] * np.conj(clock[kc]))) if 0 < kc < len(spec) else None}


def gait(runs):
    """Replay every champion on the first test terrain; keep the median-seed trace per variant."""
    out = {}
    for v, rs in runs.items():
        sigs, traces = [], {}
        for r in rs:
            cfg = r["cfg"]
            res = rollout(_champion(r), cfg, fixed_scenarios(cfg, "test")[0], record=True)
            traces[r["seed"]] = res["leg_angles"]
            sigs.append(gait_signature(res["leg_angles"]))
        order = sorted(rs, key=lambda r: r["test_distance"])
        median_seed = order[len(order) // 2]["seed"]
        ratios = [s["ratio_to_cpg"] for s in sigs if s]
        out[v] = {"median_seed": median_seed, "trace": traces[median_seed], "ratios": ratios,
                  "locked": float(np.mean([abs(x - 1) < 0.1 for x in ratios])) if ratios else None}
    return out
