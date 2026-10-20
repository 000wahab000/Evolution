"""Stats, F1-F10 and REPORT.md for one experiment, from its results folder alone.

Only F7 (robustness) and F9 (gait) re-simulate, by replaying the saved champions.

    python scripts/analyze.py results/exp1_cpg
    python scripts/analyze.py results/smoke --no-replay
"""

import argparse
import csv
import io
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.research import analysis, figures, report  # noqa: E402
from src.research.config import SCHEMA_VERSION  # noqa: E402
from src.research.runlog import read_json, write_json, write_text  # noqa: E402
from src.research.stats import iqm  # noqa: E402


def _csv(rows):
    buf = io.StringIO()
    fields = list(dict.fromkeys(k for r in rows for k in r))  # union: exploratory rows lack verdict
    w = csv.DictWriter(buf, fieldnames=fields, restval="", lineterminator="\n")
    w.writeheader()
    w.writerows(rows)
    return buf.getvalue()


def _summary_rows(summary):
    rows = []
    for v, s in summary.items():
        row = {"variant": v, "n": s["n"]}
        for m in analysis.METRICS:
            row.update({f"{m}_iqm": s[m] and s[m]["iqm"], f"{m}_ci_lo": s[m] and s[m]["ci"][0],
                        f"{m}_ci_hi": s[m] and s[m]["ci"][1], f"{m}_median": s[m] and s[m]["median"]})
        f, e = s["fall_pooled"], s["episodes_to_tau"]
        row.update({"falls": f["falls"], "test_episodes": f["episodes"], "fall_rate": f["rate"],
                    "fall_wilson_lo": f["wilson"][0], "fall_wilson_hi": f["wilson"][1],
                    "tau": e["tau"], "reached_tau": e["reached"], "median_episodes_to_tau": e["median"],
                    "dt2_drop": s["dt2_drop"], "exploit_flag": s["exploit_flag"]})
        rows.append(row)
    return rows


def _repro(exp_dir, exp, runs):
    manifest = read_json(os.path.join(exp_dir, "manifest.json"))
    metas = [r["meta"] for rs in runs.values() for r in rs]
    m0 = metas[0]
    name = exp.name
    return {
        "git_commit": f"{manifest.get('git_commit')} (dirty: {manifest.get('git_dirty')})",
        "config_sha256": exp.sha256(),
        "seeds": f"{exp.seeds}",
        "runs_finished": f"{len(metas)} of {len(exp.variants) * len(exp.seeds)}",
        "platform": m0["platform"], "cpu": m0["cpu"],
        "versions": ", ".join(f"{k} {v}" for k, v in m0["versions"].items() if v),
        "wall_time_core_h": f"{sum(m['wall_s'] for m in metas) / 3600:.2f}",
        "episodes_per_run": f"{m0['episodes_used']} train + {m0['eval_episodes']} eval",
        "terrain_version": m0["terrain_version"],
        "commands": f"python scripts/run_experiment.py experiments/{name}.json && "
                    f"python scripts/analyze.py results/{name}",
        "analyzed_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("exp_dir")
    ap.add_argument("--no-replay", action="store_true", help="skip F7/F9 (they replay champions)")
    args = ap.parse_args(argv)

    exp, runs = analysis.load(args.exp_dir)
    if not runs:
        sys.exit(f"no finished runs in {args.exp_dir}")
    rng = np.random.default_rng(0)  # bootstrap resampling only
    out = os.path.join(args.exp_dir, "summary")
    fig_dir = os.path.join(out, "figures")
    sty = figures.style([v for v in exp.variants if v in runs])

    summary = analysis.summarize_experiment(exp, runs, rng)
    rows, confirm = analysis.pairwise(exp, runs, rng)
    curves = analysis.learning_curves(exp, runs, rng)
    robust = gaits = None
    if not args.no_replay:
        print("replaying champions for F7/F9 ...")
        robust, gaits = analysis.robustness(runs), analysis.gait(runs)
    caps = report.captions(summary, confirm, gaits, replayed=not args.no_replay)

    write_json(os.path.join(out, "summary.json"), {
        "schema_version": SCHEMA_VERSION, "experiment": exp.name, "variants": summary, "style": sty,
        "curves": curves, "pairwise": rows, "confirmatory": confirm, "captions": caps,
        "per_seed": {v: [{"seed": r["seed"], "test_distance": r["test_distance"],
                          "train_distance": r["train_distance"]} for r in rs] for v, rs in runs.items()},
        "robustness": robust and {v: {k: iqm(m, axis=0).tolist() for k, m in d.items()} for v, d in robust.items()},
        "gait": gaits and {v: {"median_seed": g["median_seed"], "locked": g["locked"],
                               "median_ratio": float(np.median(g["ratios"])) if g["ratios"] else None}
                           for v, g in gaits.items()},
    })
    write_text(os.path.join(out, "summary.csv"), _csv(_summary_rows(summary)))
    write_text(os.path.join(out, "pairwise.csv"), _csv(rows + confirm) if rows or confirm else "")

    draws = {"F1": figures.f1_curves(curves, sty), "F2": figures.f2_final(runs, sty, rng),
             "F3": figures.f3_gap(runs, sty), "F4": figures.f4_profile(runs, sty),
             "F5": figures.f5_matrix(rows, sty), "F6": figures.f6_efficiency(runs, sty),
             "F8": figures.f8_tradeoff(runs, sty), "F10": figures.f10_falls(summary, sty)}
    if robust:
        draws["F7"] = figures.f7_robustness(robust, sty, rng)
        draws["F9"] = figures.f9_gait(gaits, sty)
    for name, draw in draws.items():
        figures.save_all(draw, name, fig_dir)

    write_text(os.path.join(out, "REPORT.md"),
               report.build(exp, summary, rows, confirm, caps, _repro(args.exp_dir, exp, runs)))
    for c in confirm:
        print(f"{c['id']}: {c['a']} vs {c['b']} on {c['metric']} -> {c['verdict']}")
    print(f"wrote {os.path.join(out, 'REPORT.md')}")


if __name__ == "__main__":
    main()
