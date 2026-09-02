---
name: Evolution-1
last_updated: 2026-09-01
---

# Evolution-1 Strategy

## Purpose

Complex continuous motor control and artificial life are notoriously difficult to explore without heavyweight reinforcement learning infrastructure and opaque reward tuning. Evolution-1 exists to make physical locomotion, morphology evolution, and multi-agent dynamics transparent, fast to iterate, and understandable from first principles.

## Positioning

A lightweight, zero-bloat 2D physics evolutionary laboratory combining Central Pattern Generators (CPGs), direct multi-core genetic algorithms, and real-time browser telemetry without heavy ML dependencies.

## Users

**Primary:** Autonomous systems & AI researcher - They are hiring Evolution-1 to rapidly test and visualize emergent locomotion, morphology adaptation, and competitive multi-agent dynamics.

## Boundaries

- No heavy, slow deep RL training frameworks (e.g. PyTorch RL / Ray / Stable-Baselines) when lightweight genetic algorithms and CPGs solve the gait efficiently.
- No generic AI-slop UI aesthetics (glowing purple gradients, aurora blobs, blurred drop-shadow cards). Interface remains industrial, high-contrast, brutalist, and data-dense.

_Resist a change when:_ It adds complex black-box abstractions or slows down rollout throughput for negligible gait improvement.

## Key metrics

- **Forward Displacement (px)** - Torso x-axis displacement achieved before episode termination (measured in physics simulation).
- **Fall Termination Rate (%)** - Percentage of population rollouts terminated early due to torso ground contact or severe tilt.
- **Population Diversity (σ)** - Standard deviation across neural network weights in the active population to monitor genetic stagnation.
- **Evaluation Throughput (genomes/s)** - Number of simulated episodes executed per second across multi-core CPU workers.

## Tracks

### Locomotion & Morphology Pipeline

Evolving physics agents from basic quadruped forward walking (v1-v2) to procedural obstacle navigation (v3), balanced horse-and-rider morphologies (v4), and spear-wielding vision agents (v5).

_Why it serves the approach:_ Progressively increases behavioral and mechanical complexity on a verified physics baseline.

### High-Throughput Evolutionary Engine

Optimizing parallel rollout execution, headless state serialization, and multi-core multiprocessing to keep training iterations under seconds.

_Why it serves the approach:_ Fast iteration speed allows rapid hypothesis testing and immediate visual feedback.

### Multi-Agent Battle Arena

Building competitive multi-creature physics environments (v6) where agents evolve combat tactics, balance retention under collision, and target engagement.

_Why it serves the approach:_ Replaces static reward engineering with open-ended competitive co-evolution.

## Brand

**One-liner:** Lightweight evolutionary physics lab for emergent quadruped locomotion and competitive artificial life.
