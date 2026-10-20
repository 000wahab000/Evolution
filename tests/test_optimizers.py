import numpy as np
import pytest

from src.research.config import make_rng
from src.research.optimizers import OPTIMIZERS, GAOptimizer, make_optimizer

SMALL = {"ga": {"pop_size": 20}, "cmaes": {}, "openai_es": {"pop_size": 20, "sigma": 0.1, "lr": 0.05},
         "random_search": {"n_directions": 8, "step_size": 0.05, "noise": 0.1}}


def sphere(x):
    return -float(np.sum(np.asarray(x) ** 2))


def run(name, budget=2000, seed=0):
    opt = make_optimizer(name, 10, SMALL[name], make_rng(seed, name))
    x0 = np.full(10, 1.0)
    if hasattr(opt, "theta"):
        opt.theta = x0.copy()
    used, history = 0, []
    while True:
        genomes = opt.ask()
        if used + len(genomes) > budget:
            break
        fits = [sphere(g) for g in genomes]
        used += len(genomes)
        history.append(max(fits))
        opt.tell(fits)
    return used, history


@pytest.mark.parametrize("name", sorted(OPTIMIZERS))
def test_improves_on_sphere(name):
    used, history = run(name)
    assert used <= 2000
    assert max(history[-3:]) > history[0] + 0.5 * abs(history[0])


@pytest.mark.parametrize("name", sorted(OPTIMIZERS))
def test_deterministic_per_seed(name):
    assert run(name, 400, seed=3) == run(name, 400, seed=3)
    assert run(name, 400, seed=3) != run(name, 400, seed=4)


def test_antithetic_pairs_count_as_two():
    for name in ("openai_es", "random_search"):
        opt = make_optimizer(name, 10, SMALL[name], make_rng(0))
        g = opt.ask()
        n = len(g) // 2
        assert len(g) == {"openai_es": 20, "random_search": 16}[name]
        assert np.allclose(g[0] - opt.theta, -(g[n] - opt.theta))


def test_ga_keeps_elites_and_size():
    ga = GAOptimizer(10, make_rng(0), pop_size=20)
    pop = ga.ask()
    fits = list(range(20))
    ga.tell(fits)
    new = ga.ask()
    assert len(new) == 20
    for k, i in enumerate(np.argsort(fits)[::-1][:5]):
        assert (new[k] == pop[i]).all()
