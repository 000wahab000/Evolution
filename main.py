import argparse
import csv
import logging
import os
import random

import numpy as np
import pygame
import pymunk
import pymunk.pygame_util
import torch

from src.environment import CreatureEnv
from src.evolution import Population
from src.model import CreatureBrain


def train(generations=100, pop_size=50, fitness_mode="baseline"):
    logging.info(
        f"Starting training for {generations} generations with population size {pop_size}..."
    )
    env = CreatureEnv(fitness_mode=fitness_mode)
    population = Population(CreatureBrain, pop_size=pop_size)

    with open("history.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["Generation", "Best_Fitness", "Average_Fitness", "Worst_Fitness", "Diversity_Score"]
        )

    best_overall_fitness = -float("inf")
    best_overall_model = None

    for gen in range(generations):
        for i, model in enumerate(population.models):
            state = env.reset()
            total_reward = 0
            done = False

            while not done:
                action = model.get_action(state)
                state, reward, done, fallen = env.step(action)
                total_reward += reward

            if fallen:
                total_reward -= 100.0

            population.fitnesses[i] = total_reward

        gen_best_fitness = max(population.fitnesses)
        gen_best_idx = population.fitnesses.index(gen_best_fitness)

        if gen_best_fitness > best_overall_fitness:
            best_overall_fitness = gen_best_fitness
            best_overall_model = population.models[gen_best_idx]
            torch.save(best_overall_model.state_dict(), "best_model.pth")

        # Save top 5 of the current generation for rendering
        sorted_indices = np.argsort(population.fitnesses)[::-1]
        for idx_rank in range(min(5, len(population.models))):
            torch.save(
                population.models[sorted_indices[idx_rank]].state_dict(), f"top_{idx_rank}.pth"
            )

        fitnesses = np.array(population.fitnesses)
        best_fit = np.max(fitnesses)
        avg_fit = np.mean(fitnesses)
        worst_fit = np.min(fitnesses)

        all_params = []
        for model in population.models:
            params = torch.cat([p.flatten() for p in model.parameters()]).detach().numpy()
            all_params.append(params)
        all_params = np.array(all_params)
        diversity = np.mean(np.std(all_params, axis=0))

        logging.info(
            f"Gen {gen:3d} | Best: {best_fit:7.2f} | Avg: {avg_fit:7.2f} | Worst: {worst_fit:7.2f} | Div: {diversity:.4f}"
        )

        with open("history.csv", "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([gen, best_fit, avg_fit, worst_fit, diversity])

        population.evolve()

    logging.info("Training complete. Best model saved to best_model.pth")


def render():
    logging.info("Rendering top 5 models...")
    if not os.path.exists("top_0.pth") and not os.path.exists("best_model.pth"):
        logging.warning("No models found. Run --train first.")
        return

    pygame.init()
    screen = pygame.display.set_mode((800, 600))
    pygame.display.set_caption("Evolved Locomotion - Top 5 Race")
    clock = pygame.time.Clock()
    draw_options = pymunk.pygame_util.DrawOptions(screen)
    font = pygame.font.SysFont(None, 24)

    envs = [CreatureEnv() for _ in range(5)]
    models = [CreatureBrain() for _ in range(5)]
    states = [env.reset() for env in envs]
    total_rewards = [0] * 5
    dones = [False] * 5

    for i in range(5):
        if os.path.exists(f"top_{i}.pth"):
            models[i].load_state_dict(torch.load(f"top_{i}.pth"))
        elif os.path.exists("best_model.pth"):
            models[i].load_state_dict(torch.load("best_model.pth"))
        models[i].eval()

    running = True

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        for i in range(5):
            if not dones[i]:
                action = models[i].get_action(states[i])
                states[i], reward, dones[i], fallen = envs[i].step(action)
                total_rewards[i] += reward

        # Camera follows the leader (max x position)
        best_x = max([env.torso.position.x for env in envs])
        tx = 400 - best_x
        draw_options.transform = pymunk.Transform.translation(tx, 0)

        screen.fill((255, 255, 255))

        # Draw Ruler
        start_x = int(-tx) - (int(-tx) % 100)
        end_x = start_x + 900

        for x in range(start_x, end_x, 100):
            screen_x = x + tx
            pygame.draw.line(screen, (200, 200, 200), (screen_x, 500), (screen_x, 520), 2)
            label = font.render(f"{x}px", True, (150, 150, 150))
            screen.blit(label, (screen_x + 5, 505))

        for env in envs:
            env.space.debug_draw(draw_options)

        # Draw Stats
        leader_idx = np.argmax([env.torso.position.x for env in envs])
        leader_env = envs[leader_idx]
        leader_action = models[leader_idx].get_action(states[leader_idx])

        stats = [
            f"--- LEADER STATS (Creature {leader_idx + 1}) ---",
            f"Speed: {leader_env.torso.velocity.x:.1f} px/s",
            f"Torso Angle: {leader_env.torso.angle:.2f} rad",
            "Motor Actions:",
        ]

        for leg_idx, a in enumerate(leader_action):
            stats.append(f"  Leg {leg_idx + 1}: {a * 100:5.1f}%")

        stats.append("")
        stats.append("--- LEADERBOARD ---")
        for i in range(5):
            stats.append(f"Creature {i + 1} Distance: {envs[i].torso.position.x - 400:.1f} px")

        stats.append("")
        stats.append(f"Steps: {max(env.steps for env in envs)} / 600")

        for i, text in enumerate(stats):
            # Change color slightly for each creature text to differentiate, or just use black
            img = font.render(text, True, (0, 0, 0))
            screen.blit(img, (20, 20 + i * 25))

        pygame.display.flip()
        clock.tick(60)

        if all(dones):
            states = [env.reset() for env in envs]
            total_rewards = [0] * 5
            dones = [False] * 5

    pygame.quit()


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler("evolution.log"), logging.StreamHandler()],
    )

    parser = argparse.ArgumentParser(description="Evolved Locomotion in Python")
    parser.add_argument("--train", action="store_true", help="Run the evolutionary training loop")
    parser.add_argument("--render", action="store_true", help="Render the best genome")
    parser.add_argument("--gens", type=int, default=100, help="Number of generations to train")
    parser.add_argument("--pop", type=int, default=50, help="Population size")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    parser.add_argument(
        "--fitness-mode",
        type=str,
        default="baseline",
        choices=["baseline", "tilt_penalized", "energy_penalized", "all_penalties"],
        help="Fitness function mode",
    )
    args = parser.parse_args()

    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        random.seed(args.seed)

    if args.train:
        train(args.gens, args.pop, args.fitness_mode)
    elif args.render:
        render()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
