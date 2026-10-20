"""Copy an analyzed experiment into the dashboard's static bundle (no server simulation).

    python scripts/analyze.py results/exp1_cpg
    python scripts/export_dashboard.py results/exp1_cpg

Writes web/static/experiments/<exp>/{index,curves,runs}.json plus dark SVG figures, the
median-seed champion replay per variant under web/static/replays/<exp>/, and updates
web/static/experiments/index.json (the experiment picker). Budget: 300 KB per experiment bundle.
"""

import argparse
import os
import shutil
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from scripts.record_replay import record_champion, save  # noqa: E402
from src.research.runlog import read_json, write_json  # noqa: E402

STATIC = os.path.join(ROOT, "web", "static")
SVG_FIGURES = ["F3", "F4", "F6", "F7", "F8", "F9", "F10"]  # F1, F2, F5 are drawn natively
BUDGET_KB = 300
CHART_DASH = {"solid": [], "dashed": [6, 4], "dashdot": [6, 3, 2, 3], "dotted": [2, 3]}


def _downsample(xs, n=40):
    idx = np.unique(np.linspace(0, len(xs) - 1, min(n, len(xs))).round().astype(int))
    return idx.tolist()


def export(exp_dir, static=STATIC, replays=True):
    s = read_json(os.path.join(exp_dir, "summary", "summary.json"))
    name = s["experiment"]
    out = os.path.join(static, "experiments", name)
    if os.path.exists(out):
        shutil.rmtree(out)
    os.makedirs(out)

    style = {v: {**st, "chartDash": CHART_DASH.get(st["dash"], [])} for v, st in s["style"].items()}
    idx = _downsample(s["curves"]["episodes"])
    curves = {"episodes": [s["curves"]["episodes"][i] for i in idx]}
    for v in style:
        c = s["curves"][v]
        curves[v] = {k: [round(c[k][i], 1) for i in idx] for k in ("iqm", "lo", "hi")}

    runs = {v: [{"seed": r["seed"], "test": round(r["test_distance"], 1), "train": round(r["train_distance"], 1)}
                for r in rows] for v, rows in s["per_seed"].items()}

    replay_paths = {}
    if replays:
        for v, rows in s["per_seed"].items():
            median = sorted(rows, key=lambda r: r["test_distance"])[len(rows) // 2]["seed"]
            champ = os.path.join(exp_dir, v, f"seed_{median}", "champion.npz")
            rel = f"{name}/{v}_median"
            save(record_champion(champ, f"{name} / {v} / median seed {median} champion"),
                 os.path.join(static, "replays", f"{rel}.json"))
            replay_paths[v] = rel

    figs = {}
    for f in SVG_FIGURES:
        src = os.path.join(exp_dir, "summary", "figures", f"{f}.dark.svg")
        if os.path.exists(src):
            shutil.copy(src, os.path.join(out, f"{f}.svg"))
            figs[f] = f"{f}.svg"

    summary_rows = [{
        "variant": v, "n": d["n"],
        "test": d["test_distance"], "gap_test": d["gap_test"] and d["gap_test"]["iqm"],
        "gap_control": d["gap_control"] and d["gap_control"]["iqm"],
        "fall_rate": d["fall"]["mean"], "fall_ci": d["fall"]["ci"],
        "cot": d["cot"] and d["cot"]["iqm"], "episodes_to_tau": d["episodes_to_tau"],
        "dt2_drop": d["dt2_drop"], "dt2_flagged": d["dt2_flagged"], "exploit_flag": d["exploit_flag"],
    } for v, d in s["variants"].items()]
    pairs = [{k: r[k] for k in ("a", "b", "cliffs_delta", "p_improvement", "p_holm")} for r in s["pairwise"]]
    verdicts = [{k: r[k] for k in ("id", "a", "b", "metric", "direction", "iqm_diff", "diff_ci_lo",
                                   "diff_ci_hi", "p_holm", "verdict")} for r in s["confirmatory"]]
    write_json(os.path.join(out, "index.json"), {
        "experiment": name, "exported_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "style": style, "summary": summary_rows, "verdicts": verdicts, "pairs": pairs,
        "captions": s["captions"], "figures": figs, "replays": replay_paths,
    })
    write_json(os.path.join(out, "curves.json"), curves)
    write_json(os.path.join(out, "runs.json"), runs)

    listing_path = os.path.join(static, "experiments", "index.json")
    listing = read_json(listing_path) if os.path.exists(listing_path) else {"experiments": []}
    listing["experiments"] = [e for e in listing["experiments"] if e["name"] != name]
    listing["experiments"].insert(0, {"name": name, "exported_utc": time.strftime("%Y-%m-%d")})
    write_json(listing_path, listing)

    size_kb = sum(os.path.getsize(os.path.join(out, f)) for f in os.listdir(out)) / 1024
    return name, size_kb


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("exp_dir")
    ap.add_argument("--no-replays", action="store_true", help="skip recording median champions")
    args = ap.parse_args(argv)
    name, kb = export(args.exp_dir, replays=not args.no_replays)
    print(f"exported {name}: {kb:.0f} KB" + (f"  (over the {BUDGET_KB} KB budget!)" if kb > BUDGET_KB else ""))
    return 0 if kb <= BUDGET_KB else 1


if __name__ == "__main__":
    sys.exit(main())
