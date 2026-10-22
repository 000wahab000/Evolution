"""One small GA run for the playground notebook (a module so worker processes can import it)."""

import numpy as np

from src.research.config import RunConfig, make_rng
from src.research.evaluate import DIM, evaluate_genome, fixed_scenarios, train_scenarios
from src.research.optimizers import make_optimizer


def make_cfg(cpg, seed):
    return RunConfig(experiment="playground", variant="cpg" if cpg else "no_cpg", seed=seed,
                     brain_input="cpg" if cpg else "zeros", rollouts_k=2)


def evolve(args):
    """args = (cpg, mutation_rate, generations, population, seed) -> (test distance, genome list)."""
    cpg, mutation_rate, generations, population, seed = args
    cfg = make_cfg(cpg, seed)
    opt = make_optimizer("ga", DIM, {"pop_size": population, "mutation_rate": mutation_rate},
                         make_rng(seed, "optimizer"))
    rng = make_rng(seed, "train")
    for _ in range(generations):
        scenarios = train_scenarios(cfg, rng)  # same terrains for the whole generation
        opt.tell([evaluate_genome(g, cfg, scenarios)[0] for g in opt.ask()])
    champ = opt.recommend()
    test = evaluate_genome(champ, cfg, fixed_scenarios(cfg, "test"))[1]  # terrains never trained on
    return float(np.mean([r["distance"] for r in test])), champ.tolist()
