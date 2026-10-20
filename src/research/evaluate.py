"""Rollouts on sampled terrains with seeded perturbations, and the raw metrics per episode.

A scenario fixes everything random about one episode (terrain, friction, pose noise, CPG
noise seed), so all genomes of a generation can share the same k scenarios.
"""

from dataclasses import dataclass

import numpy as np

from src.environment import CreatureEnv
from src.model import CreatureBrain
from src.research.config import make_rng
from src.terrain import legacy_course, random_course

SPAWN_X = 400.0
SPLITS = ("train", "validation", "test", "control")
SHAPES = {"w1": (16, 15), "b1": (16,), "w2": (4, 16), "b2": (4,)}
DIM = sum(int(np.prod(s)) for s in SHAPES.values())  # 324


def to_flat(state_dict):
    return np.concatenate([np.asarray(state_dict[k], dtype=np.float64).ravel() for k in SHAPES])


def from_flat(vec):
    out, i = {}, 0
    for k, shape in SHAPES.items():
        n = int(np.prod(shape))
        out[k] = np.asarray(vec[i:i + n], dtype=np.float32).reshape(shape)
        i += n
    return out


def init_genome(rng):
    """Same distribution as CreatureBrain(): N(0, 0.25) for every weight and bias."""
    return rng.normal(0.0, 0.25, DIM)


@dataclass(frozen=True)
class Scenario:
    terrain: object
    friction_scale: float = 1.0
    pose_noise: tuple = None
    cpg_seed: int = 0


class EpisodeCounter:
    def __init__(self):
        self.n = 0

    def add(self, k=1):
        self.n += k


def split_terrain(cfg, split, terrain_seed):
    """Terrain seeds of different splits come from disjoint SeedSequence streams."""
    t = cfg.terrain
    if t.get("kind") == "legacy":
        return legacy_course()
    rng = np.random.default_rng(np.random.SeedSequence([SPLITS.index(split), int(terrain_seed)]))
    lo, hi = t["test_slope"] if split == "test" else (0.0, t["max_slope"])
    return random_course(rng, hi, length=t.get("length", 3000), n_segments=t.get("n_segments", 10),
                         min_slope=lo, name=f"{split}_{terrain_seed}")


def train_scenarios(cfg, rng):
    """k fresh train scenarios with friction x U[lo, hi] and N(0, pose_sigma) leg angles."""
    p = cfg.perturbations
    out = []
    for _ in range(cfg.rollouts_k):
        terrain = split_terrain(cfg, "train", int(rng.integers(2**31)))
        friction = float(rng.uniform(*p["friction"])) if p.get("friction") else 1.0
        sigma = p.get("pose_sigma", 0.0)
        pose = tuple(rng.normal(0.0, sigma, 4)) if sigma else None
        out.append(Scenario(terrain, friction, pose, int(rng.integers(2**31))))
    return out


def fixed_scenarios(cfg, split):
    """Held-out sets: nominal physics, one scenario per configured terrain seed."""
    seeds = getattr(cfg, f"{split}_terrain_seeds")
    return [Scenario(split_terrain(cfg, split, s), cpg_seed=int(s)) for s in seeds]


def rollout(params, brain_input, scenario, max_steps=600, substeps=1, record=False):
    """One episode. Returns distance, fell, steps, energy (sum |action|), cot, return (the
    training reward incl. -100 on fall), plus leg angles / CPG phase when record=True."""
    brain = CreatureBrain()
    brain.load_state_dict(params if isinstance(params, dict) else from_flat(params))
    env = CreatureEnv(terrain=scenario.terrain.scaled_friction(scenario.friction_scale),
                      cpg_mode=brain_input, cpg_rng=make_rng(scenario.cpg_seed, "cpg"),
                      substeps=substeps)
    state = env.reset(pose_noise=scenario.pose_noise)
    ret, energy, steps, done, fallen, legs = 0.0, 0.0, 0, False, False, []
    while not done and steps < max_steps:
        action = brain.get_action(state)
        state, reward, done, fallen = env.step(action)
        ret += reward
        energy += float(np.abs(action).sum())
        steps += 1
        if record:
            legs.append([leg.angle - env.torso.angle for leg in env.legs])
    if fallen:
        ret -= 100.0
    distance = float(env.torso.position.x - SPAWN_X)
    out = {"distance": distance, "fell": bool(fallen), "steps": steps, "energy": energy,
           "cot": energy / distance if distance > 0 else float("nan"), "return": ret}
    if record:
        out["leg_angles"] = np.array(legs)
        out["cpg_phase"] = np.arange(1, steps + 1) % 30
    return out


def evaluate_genome(params, cfg, scenarios, counter=None, substeps=1):
    """Mean training reward over the scenarios plus per-rollout metrics.
    Pass counter only for train-split rollouts; held-out evals never touch the budget."""
    rolls = [rollout(params, cfg.brain_input, s, cfg.max_steps, substeps) for s in scenarios]
    if counter is not None:
        counter.add(len(rolls))
    return float(np.mean([r["return"] for r in rolls])), rolls


def summarize(rolls):
    """M1 distance, M2 fall rate, M3 cost of transport (NaN-safe mean over distance > 0)."""
    cots = [r["cot"] for r in rolls if r["distance"] > 0]
    return {"distance": float(np.mean([r["distance"] for r in rolls])),
            "fall_rate": float(np.mean([r["fell"] for r in rolls])),
            "falls": int(sum(r["fell"] for r in rolls)), "episodes": len(rolls),
            "cot": float(np.mean(cots)) if cots else float("nan")}
