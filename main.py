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
    mono_name = "Consolas" if pygame.font.match_font("Consolas") else None
    hud_font = pygame.font.SysFont(mono_name, 14)
    hud_bold = pygame.font.SysFont(mono_name, 14, bold=True)
    hud_title_font = pygame.font.SysFont(mono_name, 15, bold=True)

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

    # Latency telemetry state (EMA smoothed)
    latencies = {
        "NN Inference": 0.1,
        "Physics Step": 0.5,
        "Pymunk Render": 10.0,
        "UI / Overlays": 0.5,
        "Display / Flip": 1.0,
    }
    total_frame_ms = 16.6
    alpha = 0.08

    while running:
        t_frame_start = time.perf_counter()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # Phase 1: Neural Network Inference
        t0 = time.perf_counter()
        actions = [None] * 5
        for i in range(5):
            if not dones[i]:
                actions[i] = models[i].get_action(states[i])
        t_nn = (time.perf_counter() - t0) * 1000.0

        # Phase 2: Physics Step
        t0 = time.perf_counter()
        for i in range(5):
            if not dones[i] and actions[i] is not None:
                states[i], _, dones[i], _ = envs[i].step(actions[i])
        t_phys = (time.perf_counter() - t0) * 1000.0

        # Camera follows the leader (max x position)
        best_x = max([env.torso.position.x for env in envs])
        tx = 400 - best_x
        draw_options.transform = pymunk.Transform.translation(tx, 0)

        screen.fill((255, 255, 255))

        # Phase 3: Pymunk Space Render
        t0 = time.perf_counter()
        for env in envs:
            env.space.debug_draw(draw_options)
        t_render = (time.perf_counter() - t0) * 1000.0

        # Phase 4: UI & Overlays (Ruler + Leader Stats + Latency HUD)
        t0 = time.perf_counter()

        # Draw Ruler
        start_x = int(-tx) - (int(-tx) % 100)
        end_x = start_x + 900

        for x in range(start_x, end_x, 100):
            screen_x = x + tx
            pygame.draw.line(screen, (200, 200, 200), (screen_x, 500), (screen_x, 520), 2)
            label = font.render(f"{x}px", True, (150, 150, 150))
            screen.blit(label, (screen_x + 5, 505))

        # Draw Left Panel 1: LEADER TELEMETRY
        leader_idx = np.argmax([env.torso.position.x for env in envs])
        leader_env = envs[leader_idx]
        leader_action = actions[leader_idx] if actions[leader_idx] is not None else [0.0] * 4

        l1_w, l1_h = 285, 215
        l1_x, l1_y = 15, 15
        l1_surf = pygame.Surface((l1_w, l1_h), pygame.SRCALPHA)
        l1_surf.fill((18, 18, 24, 220))
        pygame.draw.rect(l1_surf, (55, 55, 75), (0, 0, l1_w, l1_h), width=1, border_radius=6)
        screen.blit(l1_surf, (l1_x, l1_y))

        # Title & Record
        screen.blit(hud_title_font.render(f"LEADER STATS (Creature {leader_idx + 1})", True, (255, 255, 255)), (l1_x + 12, l1_y + 10))
        screen.blit(hud_font.render(f"Universal Record: {univ_score} pts", True, (56, 189, 248)), (l1_x + 12, l1_y + 30))
        pygame.draw.line(screen, (60, 60, 80), (l1_x + 10, l1_y + 50), (l1_x + l1_w - 10, l1_y + 50), 1)

        # Leader metrics
        screen.blit(hud_font.render(f"Speed:       {leader_env.torso.velocity.x:6.1f} px/s", True, (200, 205, 215)), (l1_x + 12, l1_y + 56))
        screen.blit(hud_font.render(f"Torso Angle: {leader_env.torso.angle:6.2f} rad", True, (200, 205, 215)), (l1_x + 12, l1_y + 76))
        screen.blit(hud_bold.render("Motor Actions:", True, (255, 255, 255)), (l1_x + 12, l1_y + 96))

        for leg_idx, a in enumerate(leader_action):
            leg_y = l1_y + 115 + leg_idx * 19
            screen.blit(hud_font.render(f"  Leg {leg_idx + 1}: {a * 100:5.1f}%", True, (180, 185, 195)), (l1_x + 12, leg_y))
            # Mini motor activity bar
            bar_w = 70
            bar_x = l1_x + 195
            pygame.draw.rect(screen, (35, 35, 45), (bar_x, leg_y + 3, bar_w, 8), border_radius=2)
            fill_w = int(max(0.0, min(1.0, abs(a))) * bar_w)
            bar_color = (34, 197, 94) if a >= 0 else (239, 68, 68)
            pygame.draw.rect(screen, bar_color, (bar_x, leg_y + 3, fill_w, 8), border_radius=2)

        # Leader distance readout
        pygame.draw.line(screen, (60, 60, 80), (l1_x + 10, l1_y + 192), (l1_x + l1_w - 10, l1_y + 192), 1)
        dist_text = f"Distance: {leader_env.torso.position.x - 400:.1f} px"
        screen.blit(hud_bold.render(dist_text, True, (34, 197, 94)), (l1_x + 12, l1_y + 196))

        # Draw Left Panel 2: RACE LEADERBOARD
        l2_w, l2_h = 285, 195
        l2_x, l2_y = 15, 240
        l2_surf = pygame.Surface((l2_w, l2_h), pygame.SRCALPHA)
        l2_surf.fill((18, 18, 24, 220))
        pygame.draw.rect(l2_surf, (55, 55, 75), (0, 0, l2_w, l2_h), width=1, border_radius=6)
        screen.blit(l2_surf, (l2_x, l2_y))

        # Title & Step progress
        max_steps = max(env.steps for env in envs)
        screen.blit(hud_title_font.render("RACE LEADERBOARD", True, (255, 255, 255)), (l2_x + 12, l2_y + 10))
        screen.blit(hud_font.render(f"Step: {max_steps:3d} / 600", True, (160, 165, 175)), (l2_x + 12, l2_y + 30))

        pbar_w = 110
        pbar_x = l2_x + 155
        pygame.draw.rect(screen, (35, 35, 45), (pbar_x, l2_y + 33, pbar_w, 8), border_radius=2)
        step_fill = int((max_steps / 600.0) * pbar_w)
        pygame.draw.rect(screen, (56, 189, 248), (pbar_x, l2_y + 33, step_fill, 8), border_radius=2)

        pygame.draw.line(screen, (60, 60, 80), (l2_x + 10, l2_y + 48), (l2_x + l2_w - 10, l2_y + 48), 1)

        # Creature distances list
        for i in range(5):
            row_y = l2_y + 54 + i * 22
            is_leader = (i == leader_idx)
            c_dist = envs[i].torso.position.x - 400.0
            is_done = dones[i]

            c_color = (255, 215, 0) if is_leader else ((120, 125, 135) if is_done else (200, 205, 215))
            prefix = "> " if is_leader else "  "
            status_suffix = " (DONE)" if is_done else ""

            row_text = f"{prefix}Creature {i + 1}: {c_dist:6.1f} px{status_suffix}"
            surf = hud_bold.render(row_text, True, c_color) if is_leader else hud_font.render(row_text, True, c_color)
            screen.blit(surf, (l2_x + 8, row_y))

        # Bottom status
        pygame.draw.line(screen, (60, 60, 80), (l2_x + 10, l2_y + 170), (l2_x + l2_w - 10, l2_y + 170), 1)
        all_done = all(dones)
        status_str = "Race: In Progress" if not all_done else "Race: Complete (Resetting...)"
        status_col = (34, 197, 94) if not all_done else (234, 179, 8)
        screen.blit(hud_font.render(status_str, True, status_col), (l2_x + 12, l2_y + 175))

        # Update EMA for earlier stages
        latencies["NN Inference"] = (1 - alpha) * latencies["NN Inference"] + alpha * t_nn
        latencies["Physics Step"] = (1 - alpha) * latencies["Physics Step"] + alpha * t_phys
        latencies["Pymunk Render"] = (1 - alpha) * latencies["Pymunk Render"] + alpha * t_render

        # Determine dominant bottleneck
        bottleneck_stage, bottleneck_val = max(latencies.items(), key=lambda item: item[1])
        fps_est = 1000.0 / max(total_frame_ms, 0.001)

        # Draw Latency HUD Card (Top-Right)
        hud_w, hud_h = 285, 215
        hud_x, hud_y = 500, 15
        hud_surf = pygame.Surface((hud_w, hud_h), pygame.SRCALPHA)
        hud_surf.fill((18, 18, 24, 220))
        pygame.draw.rect(hud_surf, (55, 55, 75), (0, 0, hud_w, hud_h), width=1, border_radius=6)
        screen.blit(hud_surf, (hud_x, hud_y))

        # HUD Content
        fps_color = (34, 197, 94) if fps_est >= 50 else ((234, 179, 8) if fps_est >= 30 else (239, 68, 68))
        screen.blit(hud_title_font.render("LATENCY TELEMETRY", True, (255, 255, 255)), (hud_x + 12, hud_y + 10))

        fps_text = f"FPS: {fps_est:4.1f} ({total_frame_ms:5.1f}ms)"
        screen.blit(hud_bold.render(fps_text, True, fps_color), (hud_x + 12, hud_y + 32))

        pygame.draw.line(screen, (60, 60, 80), (hud_x + 10, hud_y + 52), (hud_x + hud_w - 10, hud_y + 52), 1)

        row_y = hud_y + 58
        for name, val in latencies.items():
            pct = (val / max(total_frame_ms, 0.001)) * 100.0
            is_bn = (name == bottleneck_stage)
            txt_color = (255, 85, 85) if is_bn else (200, 205, 215)
            prefix = "> " if is_bn else "  "

            label_surf = hud_bold.render(f"{prefix}{name}:", True, txt_color) if is_bn else hud_font.render(f"{prefix}{name}:", True, txt_color)
            val_surf = hud_bold.render(f"{val:5.2f}ms ({pct:4.1f}%)", True, txt_color) if is_bn else hud_font.render(f"{val:5.2f}ms ({pct:4.1f}%)", True, txt_color)

            screen.blit(label_surf, (hud_x + 8, row_y))
            screen.blit(val_surf, (hud_x + 155, row_y))
            row_y += 20

        # Bottleneck Callout
        pygame.draw.line(screen, (60, 60, 80), (hud_x + 10, row_y + 2), (hud_x + hud_w - 10, row_y + 2), 1)
        bn_tag = f"BOTTLENECK: {bottleneck_stage}"
        screen.blit(hud_bold.render(bn_tag[:32], True, (255, 75, 75)), (hud_x + 12, row_y + 7))

        t_ui = (time.perf_counter() - t0) * 1000.0
        latencies["UI / Overlays"] = (1 - alpha) * latencies["UI / Overlays"] + alpha * t_ui

        # Phase 5: Display Flip & VSync Tick
        t0 = time.perf_counter()
        pygame.display.flip()
        clock.tick(60)
        t_flip = (time.perf_counter() - t0) * 1000.0
        latencies["Display / Flip"] = (1 - alpha) * latencies["Display / Flip"] + alpha * t_flip

        frame_time = (time.perf_counter() - t_frame_start) * 1000.0
        total_frame_ms = (1 - alpha) * total_frame_ms + alpha * frame_time

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
