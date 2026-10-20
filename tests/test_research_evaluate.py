import numpy as np

from src.evolution import evaluate_genome as old_evaluate
from src.model import CreatureBrain
from src.research.config import RunConfig, make_rng
from src.research.evaluate import (
    DIM,
    EpisodeCounter,
    Scenario,
    evaluate_genome,
    fixed_scenarios,
    from_flat,
    to_flat,
    train_scenarios,
)
from src.terrain import legacy_course
from src.utils import UNIVERSAL_BEST_PATH, load_state_dict_file

CFG = RunConfig(experiment="t", variant="v", seed=0, rollouts_k=3, max_steps=150)


def test_flat_round_trip():
    sd = CreatureBrain().state_dict()
    vec = to_flat(sd)
    assert vec.shape == (DIM,) == (324,)
    assert all((from_flat(vec)[k] == sd[k]).all() for k in sd)


def test_legacy_k1_matches_current_reward():
    sd = load_state_dict_file(UNIVERSAL_BEST_PATH)
    cfg = RunConfig(experiment="t", variant="v", seed=0, rollouts_k=1)
    fit, rolls = evaluate_genome(sd, cfg, [Scenario(legacy_course())])
    old_reward, old_distance, old_fallen, old_steps = old_evaluate((sd, "baseline", 600))
    assert fit == old_reward
    assert rolls[0]["distance"] == old_distance and rolls[0]["steps"] == old_steps


def test_same_rng_same_result():
    vec = make_rng(1, "g").normal(0, 0.25, DIM)
    a = evaluate_genome(vec, CFG, train_scenarios(CFG, make_rng(0, "train")))
    b = evaluate_genome(vec, CFG, train_scenarios(CFG, make_rng(0, "train")))
    assert repr(a) == repr(b)  # repr: cot can be NaN, and NaN != NaN


def test_counter_counts_train_rollouts_only():
    counter, pop, gens = EpisodeCounter(), 4, 2
    rng = make_rng(0, "train")
    for _ in range(gens):
        scen = train_scenarios(CFG, rng)
        for i in range(pop):
            evaluate_genome(make_rng(i).normal(0, 0.25, DIM), CFG, scen, counter)
    evaluate_genome(np.zeros(DIM), CFG, fixed_scenarios(CFG, "test")[:2])  # held-out: no counter
    assert counter.n == pop * CFG.rollouts_k * gens


def test_test_terrains_are_steeper_than_train():
    test = fixed_scenarios(CFG, "test")[0].terrain
    s = test.segments[2]
    assert abs((s.b[1] - s.a[1]) / (s.b[0] - s.a[0])) >= 0.05 - 1e-12
