import copy
import random
import torch
import numpy as np

class Population:
    def __init__(self, model_class, pop_size=50, mutation_rate=0.1, mutation_scale=0.2):
        self.pop_size = pop_size
        self.mutation_rate = mutation_rate
        self.mutation_scale = mutation_scale
        self.models = [model_class() for _ in range(pop_size)]
        self.fitnesses = [0.0] * pop_size

    def mutate(self, model):
        child = copy.deepcopy(model)
        with torch.no_grad():
            for param in child.parameters():
                mutation_mask = (torch.rand_like(param) < self.mutation_rate).float()
                mutation_noise = torch.randn_like(param) * self.mutation_scale
                param.add_(mutation_mask * mutation_noise)
        return child

    def crossover(self, parent1, parent2):
        child = copy.deepcopy(parent1)
        with torch.no_grad():
            for p_child, p1, p2 in zip(child.parameters(), parent1.parameters(), parent2.parameters()):
                # Uniform crossover
                mask = (torch.rand_like(p1) < 0.5).float()
                p_child.copy_(mask * p1 + (1 - mask) * p2)
        return child

    def evolve(self):
        # Sort by fitness descending
        sorted_indices = np.argsort(self.fitnesses)[::-1]
        elites = [self.models[i] for i in sorted_indices[:5]] # Keep top 5

        new_models = elites[:]
        
        # Fill the rest
        while len(new_models) < self.pop_size:
            if random.random() < 0.2:
                # 20% purely mutate elites
                parent = random.choice(elites)
                new_models.append(self.mutate(parent))
            else:
                # 80% crossover then mutate
                parent1, parent2 = random.choices(elites, k=2)
                child = self.crossover(parent1, parent2)
                child = self.mutate(child)
                new_models.append(child)

        self.models = new_models
        self.fitnesses = [0.0] * self.pop_size
