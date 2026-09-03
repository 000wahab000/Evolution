# Evolution

A project for me to explore how Artificial life work in a predefined system and also be able to check out what does evolution mean

so what i have achiveed in here is that i am using pychunk to stimulate the body its just a dog at the moment with 4 legs u already know
with reward and finess scores since thats what its job is

the brain part is just a small Neutral Network with very less paramters like 256
and then its algorithm currently i just the best ones
right now it takes 5 best ones and then crossovers 30 % so that it can take a new and differnet steps and mutation is 70%

if u wanna run it
do
python main.py --render
it will run the best ones
and yeah no AI slpp btw this si what the project is about am also adding a video or gif to show you
8/26/2026

![Evolution-1 walking demo](assets/walking%20body.gif)

## 🗺️ Project Roadmap

- **✅ v1 — Straight-Line Walking (Achieved):** 2D Pymunk quadruped physics, CPG rhythm inputs, fall termination penalty, and idle reward fix to prevent reward hacking.
- **✅ v2 — Multi-Core Optimization & Web Deployment (Achieved):**
  - **Multi-Core Training:** Parallel rollout evaluation with Python multiprocessing (`--workers N`), accelerating training by 4x-8x.
  - **Live Web Dashboard:** Real-time 60 FPS HTML5 Canvas creature animation streaming over Server-Sent Events (SSE).
  - **Interactive Controls:** Start/pause evolution, tune mutation rates and population sizes dynamically from the web interface.
  - **Zero-Config Docker:** One-command deployment via `docker compose up` with volume persistence.
- **⏳ v3 — Procedural Terrain & Obstacles:** Randomly generated hills, slopes, curves, gap jumps, and spike hazards.
- **⏳ v4 — Horse Morphology & Rider:** Evolving body structure into a horse with a balanced human rider on top.
- **⏳ v5 — Weaponry & Vision:** Equipping the human rider with a spear and directional vision sensors.
- **⏳ v6 — Multi-Agent Battle Arena:** Multi-creature tournament battles with combat physics and competitive evolution.

## 🚀 Quick Start (v2)

### 1. Interactive Web Dashboard
```bash
python app.py
```
Open `http://localhost:8000` in your browser to view the live physics canvas, live convergence curves, and start/pause evolutionary training runs with one click!

### 2. Multi-Core CLI Training
```bash
# Train for 100 generations across 4 parallel CPU workers
python main.py --train --gens 100 --pop 50 --workers 4

# Render the best evolved model in PyGame
python main.py --render
```

### 3. Analytics & Progress Visualization
```bash
# Plot latest training run metrics (generates PNG & HTML dashboard)
python scripts/plot_logs.py

# Live monitor an active training run
python scripts/plot_logs.py --live
```

### 4. Docker Deployment
```bash
docker compose up --build
```

