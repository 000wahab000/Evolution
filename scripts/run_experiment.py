"""Run every (variant, seed) of an experiment config; finished runs are skipped.

    python scripts/run_experiment.py experiments/exp1_cpg.json --workers 6
    python scripts/run_experiment.py experiments/smoke.json --workers 2 --only cpg --seeds 0-1
"""

import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.research.config import ExperimentConfig  # noqa: E402
from src.research.runner import run_experiment  # noqa: E402


def parse_seeds(text):
    """'0-4' -> [0..4]; '1,3,7' -> [1, 3, 7]."""
    if not text:
        return None
    out = []
    for part in text.split(","):
        lo, _, hi = part.partition("-")
        out += list(range(int(lo), int(hi) + 1)) if hi else [int(lo)]
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config")
    ap.add_argument("--workers", type=int, default=1, help="parallel runs (one process each)")
    ap.add_argument("--only", help="comma-separated variants")
    ap.add_argument("--seeds", help="e.g. 0-4 or 1,3")
    ap.add_argument("--out", default=os.path.join(ROOT, "results"))
    args = ap.parse_args(argv)
    exp = ExperimentConfig.load(args.config)
    results = run_experiment(exp, args.out, args.workers,
                             only=args.only.split(",") if args.only else None,
                             seeds=parse_seeds(args.seeds))
    failed = [k for k, r in results.items() if r.startswith("failed")]
    print(f"{sum(r == 'done' for r in results.values())} run(s) finished, "
          f"{sum(r == 'skipped' for r in results.values())} already done, {len(failed)} failed "
          f"-> {os.path.join(args.out, exp.name)}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
