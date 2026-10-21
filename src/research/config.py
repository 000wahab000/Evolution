"""Experiment and run configs (JSON, no YAML) plus seeded RNG streams.

A run is fully determined by (RunConfig, seed). New code draws randomness only from
make_rng(); no global np.random / random.
"""

import hashlib
import json
import zlib
from dataclasses import asdict, dataclass, field, fields

import numpy as np

SCHEMA_VERSION = 1


@dataclass(frozen=True)
class RunConfig:
    experiment: str
    variant: str
    seed: int
    optimizer: str = "ga"
    optimizer_params: dict = field(default_factory=dict)
    brain_input: str = "cpg"
    episodes_budget: int = 25000
    rollouts_k: int = 5
    max_steps: int = 600
    # Train courses: |slope| <= max_slope. Test: slope magnitude in test_slope. Control: like train.
    terrain: dict = field(default_factory=lambda: {"max_slope": 0.04, "test_slope": [0.05, 0.08],
                                                   "length": 3000, "n_segments": 10})
    perturbations: dict = field(default_factory=lambda: {"friction": [0.9, 1.1], "pose_sigma": 0.05})
    eval_every_episodes: int = 1250
    test_terrain_seeds: list = field(default_factory=lambda: list(range(1000, 1010)))
    control_terrain_seeds: list = field(default_factory=lambda: list(range(2000, 2005)))
    validation_terrain_seeds: list = field(default_factory=lambda: list(range(3000, 3005)))
    # Train-distribution terrains used only to score the final champion's train distance (M4),
    # never for selection, so the overfitting gap is not biased by the champion pick.
    train_eval_terrain_seeds: list = field(default_factory=lambda: list(range(4000, 4005)))

    @property
    def run_id(self):
        return f"{self.experiment}/{self.variant}/{self.seed}"

    def to_dict(self):
        return {"schema_version": SCHEMA_VERSION, **asdict(self)}

    def sha256(self):
        return config_sha256(asdict(self))


RUN_FIELDS = {f.name for f in fields(RunConfig)} - {"experiment", "variant", "seed"}


@dataclass
class ExperimentConfig:
    name: str
    seeds: list
    variants: dict  # variant name -> dict of RunConfig overrides
    base: dict = field(default_factory=dict)  # RunConfig overrides shared by every variant
    hypotheses: list = field(default_factory=list)  # [{id, a, b, metric, direction, text}]
    hypotheses_tag: str = ""

    @classmethod
    def from_dict(cls, d):
        d = dict(d)
        version = d.pop("schema_version", SCHEMA_VERSION)
        if version != SCHEMA_VERSION:
            raise ValueError(f"unknown schema_version {version}")
        d.pop("_note", None)
        base = {k: d.pop(k) for k in list(d) if k in RUN_FIELDS}
        cfg = cls(base={**d.pop("base", {}), **base}, **d)
        for name, overrides in cfg.variants.items():
            unknown = set(cfg.base) | set(overrides or {})
            unknown -= RUN_FIELDS
            if unknown:
                raise ValueError(f"variant {name!r}: unknown config keys {sorted(unknown)}")
        return cfg

    @classmethod
    def load(cls, path):
        with open(path, encoding="utf-8") as f:
            return cls.from_dict(json.load(f))

    def to_dict(self):
        return {"schema_version": SCHEMA_VERSION, **asdict(self)}

    def sha256(self):
        return config_sha256(asdict(self))

    def resolve(self, variant, seed):
        if variant not in self.variants:
            raise KeyError(f"unknown variant {variant!r}")
        return RunConfig(experiment=self.name, variant=variant, seed=int(seed),
                         **{**self.base, **(self.variants[variant] or {})})


def config_sha256(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()


def _key(k):
    return zlib.crc32(k.encode()) if isinstance(k, str) else int(k)


def make_rng(run_seed, *keys):
    """Independent Generator per (seed, purpose); string keys are hashed to ints."""
    return np.random.default_rng(np.random.SeedSequence([int(run_seed), *map(_key, keys)]))
