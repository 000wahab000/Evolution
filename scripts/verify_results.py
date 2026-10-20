"""Check that an experiment is complete before publishing it to git. Never runs git itself.

    python scripts/verify_results.py results/exp1_cpg
"""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.research.config import ExperimentConfig  # noqa: E402
from src.research.runlog import read_json, verify_done  # noqa: E402
from src.research.runner import run_dir  # noqa: E402


def verify(exp_dir):
    exp = ExperimentConfig.from_dict(read_json(os.path.join(exp_dir, "experiment.json")))
    out_root = os.path.dirname(os.path.abspath(exp_dir))
    problems = []
    for v in exp.variants:
        for s in exp.seeds:
            problems += [f"{v}/seed_{s}: {p}" for p in verify_done(run_dir(out_root, exp.resolve(v, s)))]
    manifest = read_json(os.path.join(exp_dir, "manifest.json"))
    if manifest.get("config_sha256") != exp.sha256():
        problems.append("manifest.json config hash does not match experiment.json")
    pending = [k for k, status in manifest.get("runs", {}).items() if status != "done"]
    problems += [f"manifest: {k} not done" for k in pending]
    return problems


def main(argv=None):
    exp_dir = (argv or sys.argv[1:])[0]
    problems = verify(exp_dir)
    if problems:
        print(f"{len(problems)} problem(s):")
        print("\n".join(f"  {p}" for p in problems[:50]))
        return 1
    print("All runs complete with matching checksums. To publish this experiment once:")
    print(f"  git add -f {exp_dir.replace(os.sep, '/')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
