import numpy as np
from src.evolution import Population
from src.model import CreatureBrain


def test_population_initialization():
    pop = Population(CreatureBrain, pop_size=10)
    assert len(pop.models) == 10
    assert len(pop.fitnesses) == 10


def test_mutate_modifies_parameters():
    pop = Population(CreatureBrain, pop_size=2, mutation_rate=1.0, mutation_scale=0.5)
    model = pop.models[0]
    mutated = pop.mutate(model)

    p1 = model.parameters()[0]
    p2 = mutated.parameters()[0]
    assert not np.array_equal(p1, p2)


def test_crossover_blends_parents():
    pop = Population(CreatureBrain, pop_size=2)
    parent1 = pop.models[0]
    parent2 = pop.models[1]

    child = pop.crossover(parent1, parent2)
    assert child is not None
    for c, a, b in zip(child.parameters(), parent1.parameters(), parent2.parameters()):
        assert np.all((c == a) | (c == b))


def test_evolve_maintains_population_size():
    pop = Population(CreatureBrain, pop_size=20)
    pop.fitnesses = [float(i) for i in range(20)]
    pop.evolve()

    assert len(pop.models) == 20
    assert len(pop.fitnesses) == 20
    assert all(f == 0.0 for f in pop.fitnesses)
