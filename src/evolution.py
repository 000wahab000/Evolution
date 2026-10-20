import random

import numpy as np

# pyrefly: ignore [missing-import]
from src.environment import CreatureEnv, run_episode
# pyrefly: ignore [missing-import]
from src.model import CreatureBrain

def evaluate_genome(args):
    """
    Picklable rollout worker function for multi-core multiprocessing.
    args: tuple of (state_dict, fitness_mode, max_steps)
    returns: tuple of (total_reward, distance, fallen, steps)
    """
    state_dict, fitness_mode, max_steps = args
    brain = CreatureBrain()
    brain.load_state_dict(state_dict)
    env = CreatureEnv(fitness_mode=fitness_mode)
    total_reward, step_count, fallen = run_episode(env, brain, max_steps)
    distance = float(env.torso.position.x - 400.0)
    return total_reward, distance, fallen, step_count


class Population:
    def __init__(self, model_class=CreatureBrain, pop_size=50, mutation_rate=0.15, mutation_scale=0.35):
        self.model_class = model_class
        self.pop_size = pop_size
        self.mutation_rate = mutation_rate
        self.mutation_scale = mutation_scale
        self.models = [model_class() for _ in range(pop_size)]
        self.fitnesses = [0.0] * pop_size

    def clone_model(self, source):
        child = self.model_class()
        child.load_state_dict(source.state_dict())  # load_state_dict copies the arrays
        return child

    def evaluate(self, fitness_mode="baseline", max_steps=600, pool=None):
        tasks = [(m.state_dict(), fitness_mode, max_steps) for m in self.models]
        if pool is not None:
            results = pool.map(evaluate_genome, tasks)
        else:
            results = [evaluate_genome(task) for task in tasks]

        self.fitnesses = [r[0] for r in results]
        return results

    def mutate(self, model, in_place=False):
        child = model if in_place else self.clone_model(model)
        for param in child.parameters():
            mutation_mask = np.random.rand(*param.shape) < self.mutation_rate
            mutation_noise = np.random.randn(*param.shape) * self.mutation_scale
            param += mutation_mask * mutation_noise
        return child

    def crossover(self, parent1, parent2):
        child = self.clone_model(parent1)
        for p_child, p2 in zip(child.parameters(), parent2.parameters()):
            mask = np.random.rand(*p_child.shape) < 0.5
            p_child[:] = np.where(mask, p_child, p2)
        return child

    def evolve(self):
        sorted_indices = np.argsort(self.fitnesses)[::-1]
        elite_count = min(5, self.pop_size)
        elites = [self.models[i] for i in sorted_indices[:elite_count]]

        new_models = [self.clone_model(m) for m in elites]

        while len(new_models) < self.pop_size:
            if random.random() < 0.15:
                parent = random.choice(elites)
                new_models.append(self.mutate(parent))
            else:
                if len(elites) >= 2:
                    parent1, parent2 = random.sample(elites, 2)
                else:
                    parent1 = parent2 = elites[0]
                child = self.crossover(parent1, parent2)
                child = self.mutate(child, in_place=True)
                new_models.append(child)

        self.models = new_models
        self.fitnesses = [0.0] * self.pop_size

