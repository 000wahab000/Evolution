import multiprocessing
import pytest
import torch
import numpy as np

from src.evolution import Population, evaluate_genome
from src.model import CreatureBrain


def test_evaluate_genome_standalone():
    brain = CreatureBrain()
    state_dict = brain.state_dict()
    args = (state_dict, "baseline", 50)
    reward, distance, fallen, steps = evaluate_genome(args)

    assert isinstance(reward, float)
    assert isinstance(distance, float)
    assert isinstance(fallen, bool)
    assert isinstance(steps, int)
    assert steps <= 50


def test_population_parallel_evaluation():
    pop = Population(CreatureBrain, pop_size=6)
    with multiprocessing.Pool(processes=2) as pool:
        results = pop.evaluate(fitness_mode="baseline", max_steps=40, pool=pool)

    assert len(results) == 6
    assert len(pop.fitnesses) == 6
    for fit in pop.fitnesses:
        assert isinstance(fit, float)


def test_parallel_vs_sequential_consistency():
    torch.manual_seed(42)
    np.random.seed(42)

    pop = Population(CreatureBrain, pop_size=4)
    tasks = [(m.state_dict(), "baseline", 30) for m in pop.models]

    seq_results = [evaluate_genome(t) for t in tasks]

    with multiprocessing.Pool(processes=2) as pool:
        par_results = pool.map(evaluate_genome, tasks)

    assert len(seq_results) == len(par_results)
    for (s_rew, s_dist, s_fall, s_step), (p_rew, p_dist, p_fall, p_step) in zip(seq_results, par_results):
        assert pytest.approx(s_rew, rel=1e-3) == p_rew
        assert pytest.approx(s_dist, rel=1e-3) == p_dist
        assert s_fall == p_fall
        assert s_step == p_step
