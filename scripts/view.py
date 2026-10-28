"""
Watch an experiment run's champion live in a pygame window, on its first test terrain, looping until closed.

    python scripts/view.py                                              # knees v2 champion, seed 100
    python scripts/view.py results/pilot4_knees_v2/knees_spot/seed_101  # any run folder with champion.npz
    python scripts/view.py RUN_DIR shot.png                             # also save a screenshot at step 200

The ruler marks distance from spawn every 100px. Pins itself to 4 cores at idle priority.
"""

import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import pygame  # noqa: E402
import pymunk  # noqa: E402
import pymunk.pygame_util  # noqa: E402

from src.model import CreatureBrain  # noqa: E402
from src.research.config import RunConfig  # noqa: E402
from src.research.evaluate import fixed_scenarios, make_env  # noqa: E402
from src.utils import load_state_dict_file  # noqa: E402

try:
    import psutil

    me = psutil.Process()
    me.cpu_affinity(list(range(min(4, psutil.cpu_count()))))
    me.nice(psutil.IDLE_PRIORITY_CLASS if os.name == "nt" else 19)
except Exception:
    pass  # psutil is optional; run unpinned

run = sys.argv[1] if len(sys.argv) > 1 else "results/pilot4_knees_v2/knees/seed_100"
shot = sys.argv[2] if len(sys.argv) > 2 else None
with open(os.path.join(run, "config.json"), encoding="utf-8") as f:
    cfg_dict = json.load(f)
cfg_dict.pop("schema_version", None)
cfg = RunConfig(**cfg_dict)
env = make_env(cfg, fixed_scenarios(cfg, "test")[0])
brain = CreatureBrain()
brain.load_state_dict(load_state_dict_file(os.path.join(run, "champion.npz")))

pygame.init()
screen = pygame.display.set_mode((1000, 600))
pygame.display.set_caption(f"{cfg.experiment} / {cfg.variant} / seed {cfg.seed}")
clock = pygame.time.Clock()
font = pygame.font.SysFont(None, 24)
opts = pymunk.pygame_util.DrawOptions(screen)

running, episode = True, 0
while running:
    state, steps, done = env.reset(), 0, False
    episode += 1
    while running and not done and steps < cfg.max_steps:
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                running = False
        state, _, done, fallen = env.step(brain.get_action(state))
        steps += 1
        cam_x = env.torso.position.x - 400
        ty = 400 - env.torso.position.y
        opts.transform = pymunk.Transform.translation(-cam_x, ty)
        screen.fill((245, 245, 245))
        env.space.debug_draw(opts)
        # Ruler: a tick every 100px of distance from spawn (x=400), drawn at the ground height there
        first = int(cam_x - 400) // 100 * 100
        for d in range(first, first + 1200, 100):
            wx = d + 400
            top = None if env.terrain.legacy else env.terrain.ground_top(wx)
            gy = (495.0 if top is None else top) + ty
            sx = wx - cam_x
            pygame.draw.line(screen, (120, 120, 120), (sx, gy), (sx, gy + 20), 2)
            screen.blit(font.render(f"{d}px", True, (90, 90, 90)), (sx + 4, gy + 6))
        dist = env.torso.position.x - 400
        screen.blit(font.render(f"episode {episode}  step {steps}/{cfg.max_steps}  distance {dist:.0f}px"
                                + ("  FELL" if done else ""), True, (20, 20, 20)), (10, 10))
        pygame.display.flip()
        if shot and steps == 200:
            pygame.image.save(screen, shot)
        clock.tick(45)
pygame.quit()
