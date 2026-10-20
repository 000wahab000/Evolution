"""
Record one episode per dashboard model into web/static/replays/<name>.json.

The physics is chaotic: a 1e-6 difference in the forward pass (BLAS kernel, CPU) changes the walk,
so the server can't be trusted to reproduce what we verified locally. We record once here, commit
the files, and the dashboard plays them back client-side. Re-run after training to refresh the demo:

    python scripts/record_replay.py
    python scripts/record_replay.py --champion results/exp1_cpg/cpg/seed_7/champion.npz --out PATH

--champion replays an experiment run's champion with that run's brain input on its first test
terrain (read from the sibling config.json), so the dashboard shows what the metrics measured.
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import pymunk

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.environment import CreatureEnv  # noqa: E402
from src.model import CreatureBrain  # noqa: E402
from src.utils import MODELS_DIR, load_state_dict_file  # noqa: E402

MODELS = {"best": "best_model.npz", "universal": "universal_best.npz", "random": None}
OUT_DIR = os.path.join(ROOT, "web", "static", "replays")
MAX_STEPS = 600  # same cap as /api/stream/sim
FPS = 45  # the stream's frame rate
SPAWN_X = 400.0
FIELDS = ["tx", "ty", "ta"] + [f"l{i}{k}" for i in range(4) for k in "xya"] + ["a0", "a1", "a2", "a3", "vx"]


def terrain_polyline(env, x_min=0.0):
    """Top surface (y down) of the static ground segments, as [[x, y], ...] left to right."""
    segs = []
    for s in env.space.shapes:
        if isinstance(s, pymunk.Segment):
            a, b = s.body.local_to_world(s.a), s.body.local_to_world(s.b)
            if a.x > b.x:
                a, b = b, a
            segs.append((a.x, a.y - s.radius, b.x, b.y - s.radius))

    def surface(x, from_left):
        ys = [ay + (by - ay) * (x - ax) / (bx - ax) for ax, ay, bx, by in segs
              if (ax < x <= bx if from_left else ax <= x < bx)]
        return min(ys) if ys else None

    # ponytail: envelope sampled at segment endpoints only; crossings between them (about 1 px
    # near x=625 here) are skipped. Add pairwise intersections if terrain gets curvier.
    points = []
    for x in sorted({max(x_min, v) for ax, _, bx, _ in segs for v in (ax, bx)}):
        for y in (surface(x, True), surface(x, False)):
            if y is not None and (not points or points[-1] != [x, round(y, 2)]):
                points.append([x, round(y, 2)])
    return points


def record(name, state_dict=None, env=None, label=None):
    brain = CreatureBrain()
    if state_dict is not None:
        brain.load_state_dict(state_dict)

    env = env or CreatureEnv(fitness_mode="baseline")
    state = env.reset()
    frames, fitness, steps, done, fallen = [], 0.0, 0, False, False
    while not (done or steps >= MAX_STEPS):
        action = brain.get_action(state)
        state, reward, done, fallen = env.step(action)
        fitness += reward
        steps += 1
        frame = [round(env.torso.position.x, 2), round(env.torso.position.y, 2), round(env.torso.angle, 3)]
        for leg in env.legs:
            frame += [round(leg.position.x, 2), round(leg.position.y, 2), round(leg.angle, 3)]
        frame += [round(float(a), 2) for a in action] + [round(env.torso.velocity.x, 1)]
        frames.append(frame)
    if fallen:
        fitness -= 100.0  # same penalty as evaluate_genome

    terrain = terrain_polyline(env)
    return {
        "model": name,
        **({"label": label} if label else {}),
        "fitness": round(fitness, 2),
        "distance": round(env.torso.position.x - SPAWN_X, 2),
        "fallen": bool(fallen),
        "steps": steps,
        "fps": FPS,
        "spawn_x": SPAWN_X,
        "end_x": max(x for x, _ in terrain),  # the ground ends here; the env has no goal check
        "torso": [60, 20],
        "leg": [10, 40],
        "terrain": terrain,
        "recorded_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "fields": FIELDS,
        "frames": frames,
    }


def record_champion(champion_path, label=None):
    """Replay a run's champion.npz with the run's brain input on its first test terrain."""
    from src.research.config import RunConfig, make_rng
    from src.research.evaluate import fixed_scenarios

    with open(os.path.join(os.path.dirname(champion_path), "config.json"), encoding="utf-8") as f:
        cfg_dict = json.load(f)
    cfg_dict.pop("schema_version", None)
    cfg = RunConfig(**cfg_dict)
    scenario = fixed_scenarios(cfg, "test")[0]
    env = CreatureEnv(terrain=scenario.terrain, cpg_mode=cfg.brain_input,
                      cpg_rng=make_rng(scenario.cpg_seed, "cpg"))
    return record(cfg.run_id, load_state_dict_file(champion_path), env,
                  label or f"{cfg.experiment} / {cfg.variant} / seed {cfg.seed} champion")


def save(replay, path):
    assert all(len(f) == len(FIELDS) for f in replay["frames"])
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(replay, f, separators=(",", ":"))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--champion", help="an experiment run's champion.npz")
    ap.add_argument("--out", help="output JSON path (with --champion)")
    args = ap.parse_args(argv)
    if args.champion:
        replay = record_champion(args.champion)
        save(replay, args.out)
        print(f"{replay['label']}: distance {replay['distance']:.1f} px, steps {replay['steps']} -> {args.out}")
        return

    os.makedirs(OUT_DIR, exist_ok=True)
    for name, weights in MODELS.items():
        if name == "random":
            np.random.seed(0)  # reproducible untrained brain
        replay = record(name, load_state_dict_file(os.path.join(ROOT, MODELS_DIR, weights)) if weights else None)
        path = os.path.join(OUT_DIR, f"{name}.json")
        save(replay, path)
        print(
            f"{name:9s} fitness {replay['fitness']:8.2f}  distance {replay['distance']:7.2f} px  "
            f"steps {replay['steps']:3d}  fallen {replay['fallen']!s:5s}  {os.path.getsize(path) / 1024:5.1f} KB"
        )


if __name__ == "__main__":
    main()
