import copy
import random

import numpy as np
import torch

from src.environment import CreatureEnv
from src.model import CreatureBrain


def evaluate_genome(args):
    """
    Top-level picklable rollout worker function for multi-core multiprocessing.
    args: tuple of (state_dict, fitness_mode, max_steps)
    returns: tuple of (total_reward, distance, fallen, steps)
    """
    state_dict, fitness_mode, max_steps = args

    brain = CreatureBrain()
    brain.load_state_dict(state_dict)
    brain.eval()

    env = CreatureEnv(fitness_mode=fitness_mode)
    state = env.reset()
    total_reward = 0.0
    done = False
    step_count = 0

    while not done and step_count < max_steps:
        action = brain.get_action(state)
        state, reward, done, fallen = env.step(action)
        total_reward += reward
        step_count += 1

    if fallen:
        total_reward -= 100.0

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

    def evaluate(self, fitness_mode="baseline", max_steps=600, pool=None):
        """
        Evaluate all models in the population, optionally across a multiprocessing Pool.
        """
        tasks = [(m.state_dict(), fitness_mode, max_steps) for m in self.models]
        if pool is not None:
            results = pool.map(evaluate_genome, tasks)
        else:
            results = [evaluate_genome(task) for task in tasks]

        self.fitnesses = [r[0] for r in results]
        return results

    def mutate(self, model, in_place=False):
        child = model if in_place else copy.deepcopy(model)
        with torch.no_grad():
            for param in child.parameters():
                mutation_mask = (torch.rand_like(param) < self.mutation_rate).float()
                mutation_noise = torch.randn_like(param) * self.mutation_scale
                param.add_(mutation_mask * mutation_noise)
        return child

    def crossover(self, parent1, parent2):
        child = copy.deepcopy(parent1)
        with torch.no_grad():
            for p_child, p1, p2 in zip(
                child.parameters(), parent1.parameters(), parent2.parameters()
            ):
                # Uniform crossover
                mask = (torch.rand_like(p1) < 0.5).float()
                p_child.copy_(mask * p1 + (1 - mask) * p2)
        return child

    def evolve(self):
        # Sort by fitness descending
        sorted_indices = np.argsort(self.fitnesses)[::-1]
        elite_count = min(5, self.pop_size)
        elites = [self.models[i] for i in sorted_indices[:elite_count]]  # Keep top-N elites

        new_models = [copy.deepcopy(m) for m in elites]

        # Fill the rest
        while len(new_models) < self.pop_size:
            if random.random() < 0.15:
                # 15% purely mutate elites
                parent = random.choice(elites)
                new_models.append(self.mutate(parent))
            else:
                # 85% crossover then mutate in-place on the fresh child
                if len(elites) >= 2:
                    parent1, parent2 = random.sample(elites, 2)  # No replacement — always distinct
                else:
                    parent1 = parent2 = elites[0]
                child = self.crossover(parent1, parent2)
                child = self.mutate(child, in_place=True)
                new_models.append(child)

        self.models = new_models
        self.fitnesses = [0.0] * self.pop_size

