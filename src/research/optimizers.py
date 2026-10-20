"""Optimizers behind one ask/tell interface on flat genome vectors. Fitness: higher is better.

Every genome returned by ask() costs one evaluation (k rollouts), so an antithetic pair
counts as two. Each optimizer draws only from the Generator it is given. recommend() is the
policy the method would deploy: the GA's best individual, the mean for CMA-ES / OpenAI-ES / ARS.
"""

import math

import numpy as np


class GAOptimizer:
    """Population.evolve's operators and hyperparameters on an injected rng (not bit-equal to
    Population, which uses the global random / np.random)."""

    def __init__(self, dim, rng, pop_size=50, mutation_rate=0.15, mutation_scale=0.35,
                 elites=5, mutate_only=0.15, init_std=0.25):
        self.rng, self.dim = rng, dim
        self.rate, self.scale, self.mutate_only = mutation_rate, mutation_scale, mutate_only
        self.elites = min(elites, pop_size)
        self.pop = [rng.normal(0.0, init_std, dim) for _ in range(pop_size)]

    def ask(self):
        return [g.copy() for g in self.pop]

    def recommend(self):
        return self.pop[0].copy()  # after tell(): the best genome of the last generation

    def _mutate(self, g):
        mask = self.rng.random(self.dim) < self.rate
        return g + mask * self.rng.standard_normal(self.dim) * self.scale

    def tell(self, fitnesses):
        order = np.argsort(fitnesses)[::-1]
        elites = [self.pop[i] for i in order[:self.elites]]
        new = [e.copy() for e in elites]
        while len(new) < len(self.pop):
            if self.rng.random() < self.mutate_only:
                new.append(self._mutate(elites[self.rng.integers(len(elites))]))
            else:
                if len(elites) >= 2:
                    i, j = self.rng.choice(len(elites), 2, replace=False)
                else:
                    i = j = 0
                mask = self.rng.random(self.dim) < 0.5
                new.append(self._mutate(np.where(mask, elites[i], elites[j])))
        self.pop = new


class CMAESOptimizer:
    """pycma at its defaults; sigma0 has no library default, so it is the GA's mutation scale."""

    def __init__(self, dim, rng, x0=None, sigma0=0.35, popsize=None):
        import cma

        x0 = rng.normal(0.0, 0.25, dim) if x0 is None else x0
        # randn from the injected rng keeps pycma off the global np.random (seed=nan: don't reseed it)
        opts = {"randn": lambda *shape: rng.standard_normal(shape), "seed": math.nan, "verbose": -9}
        if popsize:
            opts["popsize"] = popsize
        self.es = cma.CMAEvolutionStrategy(x0, sigma0, opts)
        self._asked = None

    def ask(self):
        self._asked = self.es.ask()
        return [np.asarray(x) for x in self._asked]

    def tell(self, fitnesses):
        self.es.tell(self._asked, [-f for f in fitnesses])  # cma minimizes

    def recommend(self):
        return np.array(self.es.mean, dtype=float)


def _centered_ranks(x):
    ranks = np.empty(len(x))
    ranks[np.argsort(x)] = np.arange(len(x))
    return ranks / (len(x) - 1) - 0.5


class OpenAIESOptimizer:
    """Salimans et al. 2017: antithetic Gaussian perturbations, centered-rank shaping, Adam, L2."""

    def __init__(self, dim, rng, x0=None, pop_size=50, sigma=0.02, lr=0.01, weight_decay=0.005,
                 beta1=0.9, beta2=0.999):
        assert pop_size % 2 == 0, "antithetic sampling needs an even population"
        self.rng, self.n = rng, pop_size // 2
        self.theta = rng.normal(0.0, 0.25, dim) if x0 is None else np.array(x0, dtype=float)
        self.sigma, self.lr, self.wd, self.b1, self.b2 = sigma, lr, weight_decay, beta1, beta2
        self.m, self.v, self.t = np.zeros(dim), np.zeros(dim), 0

    def ask(self):
        self.eps = self.rng.standard_normal((self.n, len(self.theta)))
        return [self.theta + self.sigma * e for e in self.eps] + [self.theta - self.sigma * e for e in self.eps]

    def tell(self, fitnesses):
        w = _centered_ranks(np.asarray(fitnesses, dtype=float))
        grad = (w[:self.n] - w[self.n:]) @ self.eps / (2 * self.n * self.sigma) - self.wd * self.theta
        self.t += 1
        self.m = self.b1 * self.m + (1 - self.b1) * grad
        self.v = self.b2 * self.v + (1 - self.b2) * grad ** 2
        m_hat, v_hat = self.m / (1 - self.b1 ** self.t), self.v / (1 - self.b2 ** self.t)
        self.theta = self.theta + self.lr * m_hat / (np.sqrt(v_hat) + 1e-8)  # ascent

    def recommend(self):
        return self.theta.copy()


class RandomSearchOptimizer:
    """Mania et al. 2018 ARS V1: N antithetic directions, step scaled by the reward std."""

    def __init__(self, dim, rng, x0=None, n_directions=8, step_size=0.02, noise=0.03):
        self.rng, self.n = rng, n_directions
        self.theta = rng.normal(0.0, 0.25, dim) if x0 is None else np.array(x0, dtype=float)
        self.alpha, self.nu = step_size, noise

    def ask(self):
        self.delta = self.rng.standard_normal((self.n, len(self.theta)))
        return [self.theta + self.nu * d for d in self.delta] + [self.theta - self.nu * d for d in self.delta]

    def tell(self, fitnesses):
        r = np.asarray(fitnesses, dtype=float)
        sigma_r = r.std() or 1.0
        self.theta = self.theta + self.alpha / (self.n * sigma_r) * (r[:self.n] - r[self.n:]) @ self.delta

    def recommend(self):
        return self.theta.copy()


OPTIMIZERS = {"ga": GAOptimizer, "cmaes": CMAESOptimizer, "openai_es": OpenAIESOptimizer,
              "random_search": RandomSearchOptimizer}


def make_optimizer(name, dim, params, rng):
    if name not in OPTIMIZERS:
        raise ValueError(f"unknown optimizer {name!r}; choose from {sorted(OPTIMIZERS)}")
    return OPTIMIZERS[name](dim, rng, **(params or {}))
