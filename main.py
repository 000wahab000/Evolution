import argparse
import time
import os
import torch
import pygame
import pymunk
import pymunk.pygame_util

from src.environment import CreatureEnv
from src.model import CreatureBrain
from src.evolution import Population

def train(generations=100, pop_size=50):
    print(f"Starting training for {generations} generations with population size {pop_size}...")
    env = CreatureEnv()
    population = Population(CreatureBrain, pop_size=pop_size)
    
    best_overall_fitness = -float('inf')
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
                total_reward = -100.0 # Instant penalty for falling over!
                
            population.fitnesses[i] = total_reward
            
        gen_best_fitness = max(population.fitnesses)
        gen_best_idx = population.fitnesses.index(gen_best_fitness)
        
        if gen_best_fitness > best_overall_fitness:
            best_overall_fitness = gen_best_fitness
            best_overall_model = population.models[gen_best_idx]
            torch.save(best_overall_model.state_dict(), "best_model.pth")
            
        print(f"Gen {gen:3d} | best fitness: {gen_best_fitness:.3f} | all time best: {best_overall_fitness:.3f}")
        
        population.evolve()

    print("Training complete. Best model saved to best_model.pth")

def render():
    print("Rendering best model...")
    if not os.path.exists("best_model.pth"):
        print("No best_model.pth found. Run --train first.")
        return
        
    model = CreatureBrain()
    model.load_state_dict(torch.load("best_model.pth"))
    model.eval()
    
    pygame.init()
    screen = pygame.display.set_mode((800, 600))
    pygame.display.set_caption("Evolved Locomotion")
    clock = pygame.time.Clock()
    
    draw_options = pymunk.pygame_util.DrawOptions(screen)
    
    env = CreatureEnv()
    state = env.reset()
    
    # Adjust view
    draw_options.transform = pymunk.Transform.translation(400, 0)
    
    font = pygame.font.SysFont(None, 24)
    
    done = False
    running = True
    total_reward = 0
    
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                
        action = model.get_action(state)
        state, reward, done, fallen = env.step(action)
        total_reward += reward
        
        # Center view on creature
        tx = 400 - env.torso.position.x
        draw_options.transform = pymunk.Transform.translation(tx, 0)
        
        screen.fill((255, 255, 255))
        
        # Draw Ruler on the ground (Y=500)
        start_x = int(-tx) - (int(-tx) % 100)
        end_x = start_x + 900
        for x in range(start_x, end_x, 100):
            screen_x = x + tx
            pygame.draw.line(screen, (200, 200, 200), (screen_x, 500), (screen_x, 520), 2)
            label = font.render(f"{x}px", True, (150, 150, 150))
            screen.blit(label, (screen_x + 5, 505))
            
        env.space.debug_draw(draw_options)
        
        # Draw Stats
        stats = [
            f"Distance (Fitness): {total_reward:.1f}",
            f"Speed: {env.torso.velocity.x:.1f} px/s",
            f"Steps: {env.steps} / 600",
            f"Torso Angle: {env.torso.angle:.2f} rad",
            f"Motor Actions (Speed):"
        ]
        
        for i, a in enumerate(action):
            stats.append(f"  Leg {i+1}: {a*100:5.1f}%")
            
        for i, text in enumerate(stats):
            img = font.render(text, True, (0, 0, 0))
            screen.blit(img, (20, 20 + i * 25))
            
        pygame.display.flip()
        
        clock.tick(60)
        
        if done:
            state = env.reset()
            total_reward = 0

    pygame.quit()

def main():
    parser = argparse.ArgumentParser(description="Evolved Locomotion in Python")
    parser.add_argument("--train", action="store_true", help="Run the evolutionary training loop")
    parser.add_argument("--render", action="store_true", help="Render the best genome")
    parser.add_argument("--gens", type=int, default=100, help="Number of generations to train")
    parser.add_argument("--pop", type=int, default=50, help="Population size")
    args = parser.parse_args()

    if args.train:
        train(args.gens, args.pop)
    elif args.render:
        render()
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
