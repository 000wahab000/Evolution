"""Crash-safe run-folder writes and the run card (meta.json)."""

import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone

from src.research.config import SCHEMA_VERSION

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def replace_with_retry(tmp, path, tries=5):
    """os.replace, retried with backoff: Windows antivirus can briefly lock fresh files."""
    for i in range(tries):
        try:
            return os.replace(tmp, path)
        except PermissionError:
            if i == tries - 1:
                raise
            time.sleep(0.05 * 2 ** i)


def write_text(path, text):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="") as f:
        f.write(text)
    replace_with_retry(tmp, path)


def write_json(path, obj):
    write_text(path, json.dumps(obj, indent=2, sort_keys=True))


def read_json(path):
    with open(path, encoding="utf-8") as f:
        obj = json.load(f)
    if isinstance(obj, dict) and obj.get("schema_version", SCHEMA_VERSION) != SCHEMA_VERSION:
        raise ValueError(f"{path}: unknown schema_version {obj['schema_version']}")
    return obj


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def git_info():
    def git(*args):
        return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True,
                              timeout=10).stdout.strip()
    try:
        return {"git_commit": git("rev-parse", "HEAD") or "unknown",
                "git_dirty": bool(git("status", "--porcelain", "--untracked-files=no"))}
    except (OSError, subprocess.SubprocessError):
        return {"git_commit": "unknown", "git_dirty": None}


def versions():
    out = {"python": sys.version.split()[0]}
    for mod in ("numpy", "pymunk", "cma", "scipy"):
        try:
            m = __import__(mod)
            out[mod] = getattr(m, "__version__", None) or getattr(m, "version", "unknown")
        except ImportError:
            out[mod] = None
    return out


def utc_now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def write_meta(run_dir, cfg, started_utc, wall_s, episodes_used, eval_episodes, terrain_version):
    write_json(os.path.join(run_dir, "meta.json"), {
        "schema_version": SCHEMA_VERSION,
        "run_id": cfg.run_id,
        "seed": cfg.seed,
        **git_info(),
        "versions": versions(),
        "platform": platform.platform(),
        "cpu": platform.processor() or platform.machine(),
        "started_utc": started_utc,
        "ended_utc": utc_now(),
        "wall_s": round(wall_s, 2),
        "episodes_used": episodes_used,
        "eval_episodes": eval_episodes,
        "config_sha256": cfg.sha256(),
        "terrain_version": terrain_version,
    })


def write_done(run_dir, files):
    """DONE is written last; it lists a checksum for every output file."""
    write_json(os.path.join(run_dir, "DONE"), {
        "schema_version": SCHEMA_VERSION, "status": "done",
        "sha256": {f: sha256_file(os.path.join(run_dir, f)) for f in files},
    })


def is_done(run_dir):
    return os.path.exists(os.path.join(run_dir, "DONE"))


def verify_done(run_dir):
    """Return a list of problems (empty when DONE exists and every checksum matches)."""
    if not is_done(run_dir):
        return ["missing DONE"]
    problems = []
    for name, digest in read_json(os.path.join(run_dir, "DONE"))["sha256"].items():
        path = os.path.join(run_dir, name)
        if not os.path.exists(path):
            problems.append(f"missing {name}")
        elif sha256_file(path) != digest:
            problems.append(f"checksum mismatch {name}")
    return problems
