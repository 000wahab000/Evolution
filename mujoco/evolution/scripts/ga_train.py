"""Train the MuJoCo quadruped with the same GA used by the Pymunk experiments.

The GA code itself is reused from ``src.evolution.Population``.  Only the
evaluation step changes: each genome is a ``CreatureBrain`` whose inputs are
the MuJoCo observations and whose outputs are the eight motor commands.

Run from anywhere:

    python mujoco/evolution/scripts/ga_train.py --generations 30 --pop 20
    python mujoco/evolution/scripts/ga_train.py --generations 100 --pop 50 --workers 4

Outputs:

    mujoco/evolution/results/history.csv
    mujoco/evolution/models/best_mujoco.npz
"""

from __future__ import annotations

import argparse
import csv
import multiprocessing
import os
import random
import sys
from pathlib import Path

import numpy as np

# Make ``src`` importable when this file is run directly.  Append (not insert)
# so the repo's top-level ``mujoco/`` folder does not shadow the installed
# ``mujoco`` Python package.
ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.evolution import Population  # noqa: E402
from src.model import CreatureBrain  # noqa: E402

# Import the existing MuJoCo wrapper.  It already exposes the reset/step
# interface the GA needs: reset() -> observation, step(action) -> (obs, reward, done, fallen).
sys.path.insert(0, str(Path(__file__).resolve().parent))
from mujoco_env import MujocoDogEnv  # noqa: E402

if not hasattr(__import__("mujoco"), "MjModel"):
    raise ImportError(
        "The local 'mujoco/' folder or another file is shadowing the installed "
        "'mujoco' package. Run from outside the repo, or remove any mujoco.py "
        "on sys.path."
    )


def _brain_factory(obs_size: int, act_size: int):
    """Return a zero-argument factory, so the GA can build matching brains."""

    def make() -> CreatureBrain:
        return CreatureBrain(input_size=obs_size, output_size=act_size)

    return make


def evaluate_mujoco_genome(task: tuple[dict, int, str | None]) -> tuple[float, float, bool, int]:
    """Run one genome in MuJoCo and return (fitness, distance, fallen, steps)."""
    state_dict, max_steps, xml_path = task

    env = MujocoDogEnv(max_steps=max_steps, xml_path=xml_path)
    state = env.reset()

    brain = CreatureBrain(input_size=env.obs_size, output_size=env.act_size)
    brain.load_state_dict(state_dict)

    total_reward = 0.0
    steps = 0
    fallen = False

    for _ in range(max_steps):
        action = brain.get_action(state)
        state, reward, done, fallen = env.step(action)
        total_reward += reward
        steps += 1
        if done:
            break

    # Same one-time fall penalty as the Pymunk training loop.
    if fallen:
        total_reward -= 100.0

    distance = float(env.data.qpos[0] - env.start_x)
    return total_reward, distance, fallen, steps


class MujocoPopulation(Population):
    """Population that evaluates genomes in MuJoCo instead of Pymunk."""

    def __init__(self, model_class, pop_size, mutation_rate, mutation_scale, xml_path=None):
        super().__init__(model_class, pop_size, mutation_rate, mutation_scale)
        self.xml_path = xml_path

    def evaluate(self, max_steps: int = 600, pool=None):
        tasks = [(model.state_dict(), max_steps, self.xml_path) for model in self.models]
        if pool is not None:
            results = pool.map(evaluate_mujoco_genome, tasks)
        else:
            results = [evaluate_mujoco_genome(task) for task in tasks]

        self.fitnesses = [result[0] for result in results]
        return results

    def evolve(self):
        """Same elite/crossover/mutation scheme, but works for pop_size > 1.

        ``Population.evolve`` always keeps five elites.  With a population of
        four or fewer that means the whole population is cloned and nothing
        changes, so this version scales the elite count down for small runs.
        """
        sorted_indices = np.argsort(self.fitnesses)[::-1]
        elite_count = min(5, max(1, self.pop_size // 5))
        elites = [self.models[i] for i in sorted_indices[:elite_count]]

        new_models = [self.clone_model(model) for model in elites]
        while len(new_models) < self.pop_size:
            if random.random() < 0.15:
                new_models.append(self.mutate(random.choice(elites)))
            else:
                if len(elites) >= 2:
                    parent1, parent2 = random.sample(elites, 2)
                else:
                    parent1 = parent2 = elites[0]
                child = self.crossover(parent1, parent2)
                new_models.append(self.mutate(child, in_place=True))

        self.models = new_models
        self.fitnesses = [0.0] * self.pop_size


def train(
    generations: int = 30,
    pop_size: int = 20,
    workers: int = 1,
    max_steps: int = 600,
    mutation_rate: float = 0.15,
    mutation_scale: float = 0.35,
    seed: int = 0,
    xml_path: str | os.PathLike | None = None,
    out_dir: str | os.PathLike = "mujoco/evolution/results",
    model_path: str | os.PathLike = "mujoco/evolution/models/best_mujoco.npz",
) -> CreatureBrain:
    """Run the GA and save the best brain.  Returns the best model."""
    random.seed(seed)
    np.random.seed(seed)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    history_path = out_dir / "history.csv"

    xml = str(xml_path) if xml_path is not None else None
    probe = MujocoDogEnv(max_steps=1, xml_path=xml)
    print(f"Model: {xml or probe.model.namemodel or 'default'}")
    print(f"Observation size {probe.obs_size} -> 16 -> action size {probe.act_size}")

    population = MujocoPopulation(
        model_class=_brain_factory(probe.obs_size, probe.act_size),
        pop_size=pop_size,
        mutation_rate=mutation_rate,
        mutation_scale=mutation_scale,
        xml_path=xml,
    )

    pool = multiprocessing.Pool(processes=workers) if workers > 1 else None
    best_fitness = -float("inf")
    best_model = None

    try:
        with open(history_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["generation", "best_fitness", "mean_fitness", "best_distance", "best_steps"])

            for gen in range(generations):
                results = population.evaluate(max_steps=max_steps, pool=pool)

                best_idx = int(np.argmax(population.fitnesses))
                gen_best = float(population.fitnesses[best_idx])
                gen_mean = float(np.mean(population.fitnesses))
                best_distance = float(results[best_idx][1])
                best_steps = int(results[best_idx][3])

                if gen_best > best_fitness:
                    best_fitness = gen_best
                    best_model = population.models[best_idx]
                    Path(model_path).parent.mkdir(parents=True, exist_ok=True)
                    np.savez(model_path, **best_model.state_dict())

                print(
                    f"Gen {gen:3d} | best {gen_best:8.2f} | mean {gen_mean:8.2f} "
                    f"| distance {best_distance:7.2f} m | steps {best_steps:3d}"
                )
                writer.writerow([gen, gen_best, gen_mean, best_distance, best_steps])

                population.evolve()
    finally:
        if pool is not None:
            pool.close()
            pool.join()

    if best_model is None:
        best_model = population.models[0]
        np.savez(model_path, **best_model.state_dict())

    print(f"\nSaved best model to {model_path} (fitness {best_fitness:.2f})")
    return best_model


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generations", type=int, default=30)
    parser.add_argument("--pop", type=int, default=20)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=600)
    parser.add_argument("--mutation-rate", type=float, default=0.15)
    parser.add_argument("--mutation-scale", type=float, default=0.35)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--xml",
        default="mujoco/evolution/models/smoke.xml",
        help="MuJoCo model file to train on",
    )
    parser.add_argument("--out", default="mujoco/evolution/results")
    parser.add_argument("--model", default="mujoco/evolution/models/best_mujoco.npz")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    train(
        generations=args.generations,
        pop_size=args.pop,
        workers=args.workers,
        max_steps=args.max_steps,
        mutation_rate=args.mutation_rate,
        mutation_scale=args.mutation_scale,
        seed=args.seed,
        xml_path=args.xml,
        out_dir=args.out,
        model_path=args.model,
    )


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
