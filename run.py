"""
One entry point for the everyday jobs. Pins itself (and everything it starts) to 4 cores at idle priority.

    python run.py watch [RUN_DIR]          # live window of a run's champion (default: knees v2 seed 100)
    python run.py straight                 # straight-leg best model (main.py --render)
    python run.py dashboard                # web dashboard at http://localhost:8000
    python run.py train CONFIG [--only knees] [--seeds 100]   # local, one process, into results/local/
    python run.py cloud CONFIG             # train on GitHub Actions with the pushed code of this branch
"""

import argparse
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))


def pin():
    """Children inherit affinity and the idle priority class, so pinning this process pins them too."""
    try:
        import psutil

        me = psutil.Process()
        me.cpu_affinity(list(range(min(4, psutil.cpu_count()))))
        me.nice(psutil.IDLE_PRIORITY_CLASS if os.name == "nt" else 19)
    except Exception:
        print("psutil missing: running unpinned")


def py(*args):
    return subprocess.call([sys.executable, *args], cwd=ROOT)


def git(*args):
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True).stdout.strip()


def cloud(config):
    # Actions checks out the pushed branch, so local edits would silently not be trained.
    if git("status", "--porcelain"):
        sys.exit("uncommitted changes: commit them first, or the cloud trains the old code")
    if git("rev-list", "@{u}..HEAD"):
        sys.exit("unpushed commits: git push first")
    branch = git("branch", "--show-current")
    return subprocess.call(["gh", "workflow", "run", "experiment.yml", "--ref", branch,
                            "-f", f"experiment={config}"], cwd=ROOT)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="job", required=True)
    w = sub.add_parser("watch")
    w.add_argument("run_dir", nargs="?", default="results/pilot4_knees_v2/knees/seed_100")
    sub.add_parser("straight")
    sub.add_parser("dashboard")
    t = sub.add_parser("train")
    t.add_argument("config")
    t.add_argument("--only")
    t.add_argument("--seeds")
    c = sub.add_parser("cloud")
    c.add_argument("config")
    a = ap.parse_args()

    pin()
    if a.job == "watch":
        return py("scripts/view.py", a.run_dir)
    if a.job == "straight":
        return py("main.py", "--render")
    if a.job == "dashboard":
        return py("app.py")
    if a.job == "train":
        extra = (["--only", a.only] if a.only else []) + (["--seeds", a.seeds] if a.seeds else [])
        return py("scripts/run_experiment.py", a.config, "--out", "results/local", *extra)
    if a.job == "cloud":
        return cloud(a.config)


if __name__ == "__main__":
    sys.exit(main())
