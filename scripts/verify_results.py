"""Check that an experiment is complete before analyzing or publishing it. Never runs git itself.

    python scripts/verify_results.py results/exp1_cpg
"""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.research.runner import verify_experiment as verify  # noqa: E402


def main(argv=None):
    exp_dir = (argv or sys.argv[1:])[0]
    problems = verify(exp_dir)
    if problems:
        print(f"{len(problems)} problem(s):")
        print("\n".join(f"  {p}" for p in problems[:50]))
        return 1
    print("All runs complete with matching checksums and config. To publish this experiment once:")
    print(f"  git add -f {exp_dir.replace(os.sep, '/')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
