"""Aggregate statistics across seeds (Agarwal et al. 2021 style) and pairwise tests."""

import numpy as np
from scipy import stats as sps


def iqm(values, axis=None):
    """Interquartile mean: mean of the middle 50%."""
    return sps.trim_mean(np.asarray(values, dtype=float), 0.25, axis=axis)


def bootstrap_ci(values, rng, stat=iqm, n=10000, alpha=0.05):
    """Percentile bootstrap CI of stat over seeds (one task, so stratified = plain resampling)."""
    x = np.asarray(values, dtype=float)
    samples = x[rng.integers(0, len(x), (n, len(x)))]
    boot = stat(samples, axis=1)
    return float(np.quantile(boot, alpha / 2)), float(np.quantile(boot, 1 - alpha / 2))


def bootstrap_diff_ci(a, b, rng, stat=iqm, n=10000, alpha=0.05):
    """CI of stat(a) - stat(b), resampling each group independently."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    diff = stat(a[rng.integers(0, len(a), (n, len(a)))], axis=1) - \
        stat(b[rng.integers(0, len(b), (n, len(b)))], axis=1)
    return float(np.quantile(diff, alpha / 2)), float(np.quantile(diff, 1 - alpha / 2))


def mann_whitney(a, b):
    """Two-sided Mann-Whitney U p-value."""
    return float(sps.mannwhitneyu(a, b, alternative="two-sided").pvalue)


def holm(pvalues):
    """Holm-Bonferroni adjusted p-values, in the input order."""
    p = np.asarray(pvalues, dtype=float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(p) - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj.tolist()


def cliffs_delta(a, b):
    """P(a > b) - P(a < b) over all pairs, in [-1, 1]."""
    a, b = np.asarray(a, dtype=float)[:, None], np.asarray(b, dtype=float)[None, :]
    return float((np.sum(a > b) - np.sum(a < b)) / (a.size * b.size))


def prob_improvement(a, b):
    """P(X > Y) + 0.5 P(X = Y) for X from a, Y from b."""
    a, b = np.asarray(a, dtype=float)[:, None], np.asarray(b, dtype=float)[None, :]
    return float((np.sum(a > b) + 0.5 * np.sum(a == b)) / (a.size * b.size))


def wilson_ci(k, n, z=1.959963984540054):
    """Wilson score interval for a binomial proportion k/n."""
    if n == 0:
        return 0.0, 1.0
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return float(centre - half), float(centre + half)
