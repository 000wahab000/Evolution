# Evolution-1

A 2D physics lab where creatures learn to walk through evolution, not backprop.

No reinforcement learning frameworks. No PyTorch Lightning. No 40-package dependency hell. Just a genetic algorithm, a tiny neural net, and Pymunk doing all the heavy lifting.

![Quadruped walking demo](assets/walking%20body.gif)

---

## What this actually is

A quadruped creature (torso + 4 legs, all Pymunk rigid bodies) gets a small neural network brain - 15 inputs, one hidden layer of 16 neurons, 4 motor outputs. That's ~256 parameters total. No gradients. No loss functions.

Each generation, the top 5 performers survive. The rest of the population is bred from crossovers between survivors, then mutated. Repeat. After enough generations the creature figures out how to walk forward without falling over - entirely from scratch, with no demonstration data.

The training signal is dead simple: **go right, don't fall**.

### How the state space works

The 15 network inputs at each timestep:

| Input | What it is |
| --- | --- |
| `torso_angle` | Body tilt |
| `torso_angular_velocity` | How fast it's rotating |
| `torso_velocity_x/y` | Forward/vertical speed |
| `torso_height` | How far off the ground |
| `leg[0..3].angle` | Each leg's angle relative to torso |
| `leg[0..3].angular_velocity` | Each leg's rotation speed |
| `cpg_sin, cpg_cos` | Central Pattern Generator clock (30-step rhythm signal) |

The CPG is a precomputed sine/cosine table that gives the network a sense of rhythmic timing - it can phase-lock its gait to it. Without it, the creature tends to spam all motors at max speed and goes nowhere.

### Evolution loop

```
evaluate all N genomes (parallel, one rollout per core)
    ↓
rank by cumulative reward
    ↓
keep top 5 (elite carry-through)
    ↓
fill population:
  15% chance  → mutate one elite
  85% chance  → crossover two elites, then mutate result
    ↓
repeat
```

Mutation: per-parameter random mask (rate 0.15) with Gaussian noise (scale 0.35). Crossover: uniform parameter-level mixing (50/50 per weight from each parent).

---

## The terrain

Flat ground from spawn to x=600. Then:
- **Uphill** from x=600 to x=1600 (40px rise over 1000px)
- **Downhill** from x=1600 to x=2600 (40px drop over 1000px)

Goal is x=4000. The creature has to actually learn to handle slope changes, not just walk on flat ground.

---

## Quick start

```bash
pip install -r requirements.txt   # server + training (flask, gunicorn, numpy, pymunk)
pip install pygame matplotlib     # optional: local pygame viewer and plots (or: pip install -e .[viz])
```

**Run the best saved model:**
```bash
python main.py --render
```

**Train from scratch:**
```bash
python main.py --train --gens 100 --pop 50 --workers 4
```

**Live web dashboard** (60fps canvas, SSE stream, start/pause/tune from browser):
```bash
python app.py
# open http://localhost:8000
```

**Plot training history:**
```bash
python scripts/plot_logs.py
python scripts/plot_logs.py --live  # tails an active run
```

**Docker:**
```bash
docker compose up --build
```

---

## Deploy (Render)

Runs as a Render web service straight from the `Dockerfile` (runtime: Docker, free plan). Render sets `PORT` and gunicorn binds to it.

| Env var | What it does |
| --- | --- |
| `TRAINING_ENABLED` | `0` = watch-only public demo (start button is off, `/api/training/start` returns 403). `1` (default) lets you train from the dashboard. |
| `TRAINING_WORKERS` | Process count for training. Default 4, use `1` on the free tier. |

Free tier heads-up: it sleeps after ~15 min idle (~1 min cold start), and there's no persistent disk, so anything trained on the server resets on every deploy. The committed `models/` and `logs/` are what the demo shows.

---

## Stack

| Piece | Why |
| --- | --- |
| **Pymunk** | 2D rigid body physics - fast, no GPU needed |
| **NumPy** | The brain is 324 weights: two matmuls + `tanh`, plus crossover/mutation on plain arrays. No deep learning framework needed |
| **Flask + SSE** | Real-time browser streaming without WebSockets |
| **Python multiprocessing** | Parallel rollout evaluation across CPU cores |
| **Matplotlib** | Training curve visualization |

No Ray. No Stable-Baselines. No CUDA required.

---

## Fitness modes

The `--fitness-mode` flag changes what penalties get applied during rollout:

| Mode | What it does |
| --- | --- |
| `baseline` | Reward = forward displacement only |
| `tilt_penalized` | Also penalizes body angle deviation |
| `energy_penalized` | Also penalizes total motor effort |
| `ground_termination` | Episode ends immediately on torso contact |
| `all_penalties` | Tilt + energy penalties combined |

---

## Project layout

```
src/
  environment.py   - Pymunk physics world, creature construction, reward logic
  evolution.py     - Population class, crossover, mutation, rollout dispatch
  model.py         - CreatureBrain (15→16→4 tanh network)
models/            - Saved checkpoints (best_model.npz, top_0..4.npz)
logs/              - Training history CSVs and evolution.log
templates/         - Flask HTML templates for the web dashboard
scripts/           - plot_logs.py, other utilities
tests/             - pytest suite
```

---

## Roadmap

| Version | Status | What |
| --- | --- | --- |
| v1 - Straight-line walking | Done | Pymunk physics, CPG timing, GA baseline |
| v2 - Multi-core + web dashboard | Done | Parallel rollouts, SSE browser streaming, Docker |
| v3 - Procedural terrain | In progress | Continuously linked slope segments with progressive difficulty |
| v4 - Horse + rider | Planned | Extended morphology, rider balance constraints |
| v5 - Vision | Planned | Directional sensors for the rider |
| v6 - Multi-agent battles | Planned | Competitive co-evolution, combat physics |

---

## Why not just use RL?

Standard RL (PPO, SAC, etc.) works but it's overkill for this problem and hides what's actually happening. With a genetic algorithm you can watch the exact population structure evolve, inspect individual genomes, and run the whole thing on laptop CPUs with no framework overhead.

The network is deliberately tiny (~256 params). It's enough to solve the gait. Adding layers right now would just slow down the evolutionary search without making the walking better - the bottleneck is the fitness landscape, not model capacity.

---

## License

MIT

