import argparse
import csv
import json
import logging
import os
import random
import time

import numpy as np
import pygame
import pymunk
import pymunk.pygame_util
import torch

from src.environment import CreatureEnv
from src.evolution import Population
from src.model import CreatureBrain
from src.utils import (
    UNIVERSAL_BEST_PATH,
    UNIVERSAL_META_PATH,
    atomic_torch_save,
    load_universal_best_meta,
    save_universal_best,
)

import multiprocessing



def train(generations=100, pop_size=50, fitness_mode="baseline", mutation_rate=0.15, workers=None):
    os.makedirs("models", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    if workers is None:
        workers = os.cpu_count() or 4

    meta = load_universal_best_meta()
    universal_best_fitness = float(meta.get("best_fitness", -float("inf")))
    univ_info = f"{universal_best_fitness:.2f}" if universal_best_fitness != -float("inf") else "None"

    logging.info(
        f"Starting training for {generations} generations with population size {pop_size}, mutation_rate {mutation_rate}, workers {workers}... "
        f"(Universal All-Time Record: {univ_info})"
    )
    population = Population(CreatureBrain, pop_size=pop_size, mutation_rate=mutation_rate)

    history_path = os.path.join("logs", "history.csv")
    with open(history_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["Generation", "Best_Fitness", "Average_Fitness", "Worst_Fitness", "Diversity_Score"]
        )

    best_overall_fitness = -float("inf")
    best_overall_model = None

    pool = multiprocessing.Pool(processes=workers) if workers > 1 else None

    try:
        for gen in range(generations):
            population.evaluate(fitness_mode=fitness_mode, pool=pool)

            gen_best_fitness = max(population.fitnesses)
            gen_best_idx = population.fitnesses.index(gen_best_fitness)

            # Save session best
            if gen_best_fitness > best_overall_fitness:
                best_overall_fitness = gen_best_fitness
                best_overall_model = population.models[gen_best_idx]
                atomic_torch_save(best_overall_model.state_dict(), os.path.join("models", "best_model.pth"))

            # Save universal all-time best
            if gen_best_fitness > universal_best_fitness:
                universal_best_fitness = gen_best_fitness
                save_universal_best(
                    population.models[gen_best_idx],
                    gen_best_fitness,
                    gen,
                    fitness_mode,
                    pop_size,
                    mutation_rate,
                )
                logging.info(
                    f"[UNIVERSAL RECORD] NEW UNIVERSAL ALL-TIME BEST MODEL! Fitness: {gen_best_fitness:.2f} (Saved to models/universal_best.pth)"
                )

            # Save top 5 of the current generation for rendering
            sorted_indices = np.argsort(population.fitnesses)[::-1]
            for idx_rank in range(min(5, len(population.models))):
                atomic_torch_save(
                    population.models[sorted_indices[idx_rank]].state_dict(),
                    os.path.join("models", f"top_{idx_rank}.pth"),
                )

            fitnesses = np.array(population.fitnesses)
            best_fit = np.max(fitnesses)
            avg_fit = np.mean(fitnesses)
            worst_fit = np.min(fitnesses)

            with torch.no_grad():
                param_matrix = torch.stack([
                    torch.cat([p.view(-1) for p in m.parameters()]) for m in population.models
                ])
                diversity = float(torch.std(param_matrix, dim=0).mean().item())

            logging.info(
                f"Gen {gen:3d} | Best: {best_fit:7.2f} | Avg: {avg_fit:7.2f} | Worst: {worst_fit:7.2f} | Div: {diversity:.4f}"
            )

            with open(history_path, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([gen, best_fit, avg_fit, worst_fit, diversity])

            population.evolve()
    finally:
        if pool is not None:
            pool.close()
            pool.join()

    logging.info(
        f"Training complete. Session best: models/best_model.pth | Universal best: models/universal_best.pth ({universal_best_fitness:.2f})"
    )


def render(render_universal=False):
    universal_path = UNIVERSAL_BEST_PATH
    top_0_path = os.path.join("models", "top_0.pth")
    best_path = os.path.join("models", "best_model.pth")

    if render_universal and os.path.exists(universal_path):
        logging.info("Rendering UNIVERSAL ALL-TIME BEST model...")
    elif not os.path.exists(top_0_path) and not os.path.exists(best_path) and not os.path.exists(universal_path):
        logging.warning("No models found in models/. Run --train first.")
        return
    else:
        logging.info("Rendering top 5 models...")

    meta = load_universal_best_meta()
    univ_score = f"{meta.get('best_fitness', 0.0):.1f}" if meta else "N/A"

    pygame.init()
    screen = pygame.display.set_mode((800, 600))
    pygame.display.set_caption("Evolved Locomotion - Top 5 Race")
    clock = pygame.time.Clock()
    draw_options = pymunk.pygame_util.DrawOptions(screen)
    font = pygame.font.SysFont(None, 24)

    envs = [CreatureEnv() for _ in range(5)]
    models = [CreatureBrain() for _ in range(5)]
    states = [env.reset() for env in envs]
    dones = [False] * 5

    for i in range(5):
        if render_universal and os.path.exists(universal_path):
            path = universal_path
        else:
            m_top = os.path.join("models", f"top_{i}.pth")
            if os.path.exists(m_top):
                path = m_top
            elif os.path.exists(best_path):
                path = best_path
            elif os.path.exists(universal_path):
                path = universal_path
            else:
                path = None

        if path and os.path.exists(path):
            try:
                models[i].load_state_dict(torch.load(path, weights_only=True))
                models[i].eval()
            except Exception as e:
                logging.warning(f"Could not load {path}: {e}")


    running = True

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        for i in range(5):
            if not dones[i]:
                action = models[i].get_action(states[i])
                states[i], _, dones[i], _ = envs[i].step(action)

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
            f"Universal Record: {univ_score} pts",
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
            img = font.render(text, True, (0, 0, 0))
            screen.blit(img, (20, 20 + i * 25))

        pygame.display.flip()
        clock.tick(60)

        if all(dones):
            states = [env.reset() for env in envs]
            dones = [False] * 5

    pygame.quit()


def main():
    os.makedirs("logs", exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler(os.path.join("logs", "evolution.log"), encoding="utf-8"), logging.StreamHandler()],
    )

    parser = argparse.ArgumentParser(description="Evolved Locomotion in Python")
    parser.add_argument("--train", action="store_true", help="Run the evolutionary training loop")
    parser.add_argument("--render", action="store_true", help="Render the best genome")
    parser.add_argument("--universal", action="store_true", help="Render the universal all-time best model")
    parser.add_argument("--gens", type=int, default=100, help="Number of generations to train")
    parser.add_argument("--pop", type=int, default=50, help="Population size")
    parser.add_argument("--mutation-rate", type=float, default=0.15, help="Mutation rate for GA weights")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    parser.add_argument("--workers", type=int, default=None, help="Number of parallel worker processes for training rollouts")
    parser.add_argument(
        "--fitness-mode",
        type=str,
        default="baseline",
        choices=["baseline", "tilt_penalized", "energy_penalized", "all_penalties", "ground_termination"],
        help="Fitness function mode",
    )
    args = parser.parse_args()

    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        random.seed(args.seed)

    if args.train:
        train(args.gens, args.pop, args.fitness_mode, args.mutation_rate, args.workers)
    elif args.render or args.universal:
        render(render_universal=args.universal)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
