import numpy as np
import pytest

from src.research.stats import (
    bootstrap_ci,
    bootstrap_diff_ci,
    cliffs_delta,
    holm,
    iqm,
    mann_whitney,
    prob_improvement,
    wilson_ci,
)


def test_iqm_known_answer():
    assert iqm(range(1, 9)) == 4.5


def test_iqm_at_experiment_size():
    # n=30 (seeds per variant): trims int(0.25 * 30) = 7 from each end, averages the middle 16.
    x = np.random.default_rng(3).normal(100, 40, 30)
    assert iqm(x) == pytest.approx(np.sort(x)[7:23].mean())
    assert iqm(list(range(30)) + [10_000]) < 20  # one outlier seed barely moves it


def test_bootstrap_of_constant_is_constant():
    assert bootstrap_ci([3.0] * 10, np.random.default_rng(0), n=500) == (3.0, 3.0)
    assert bootstrap_diff_ci([3.0] * 5, [1.0] * 5, np.random.default_rng(0), n=500) == (2.0, 2.0)


def test_bootstrap_ci_brackets_iqm():
    x = np.random.default_rng(1).normal(10, 2, 30)
    lo, hi = bootstrap_ci(x, np.random.default_rng(0), n=2000)
    assert lo < iqm(x) < hi


def test_cliffs_delta_and_prob_improvement():
    assert cliffs_delta([5, 6, 7], [1, 2, 3]) == 1.0
    assert cliffs_delta([1, 2, 3], [5, 6, 7]) == -1.0
    assert cliffs_delta([1, 2], [1, 2]) == 0.0
    assert prob_improvement([5, 6], [1, 2]) == 1.0
    assert prob_improvement([1, 2], [1, 2]) == 0.5


def test_holm_textbook():
    # Holm (1979) style example: sorted p 0.01..0.04, m=4 -> 0.04, 0.06, 0.06, max(0.04, 0.06).
    assert holm([0.04, 0.01, 0.03, 0.02]) == pytest.approx([0.06, 0.04, 0.06, 0.06])
    assert holm([0.5, 0.9]) == pytest.approx([1.0, 1.0])  # running max


def test_mann_whitney_separated_samples():
    assert mann_whitney(range(10), range(100, 110)) < 0.001
    assert mann_whitney([1, 2, 3], [1, 2, 3]) == 1.0


def test_wilson_reference_value():
    # 8 of 10, 95%: (0.4902, 0.9433) per Wilson (1927) / standard tables.
    lo, hi = wilson_ci(8, 10)
    assert (round(lo, 4), round(hi, 4)) == (0.4902, 0.9433)
