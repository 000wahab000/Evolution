# Shared Domain Concepts & Vocabulary

This document outlines the domain vocabulary, biological metaphors, and architectural mechanisms used in the Pod Pedal Walker evolutionary simulation.

## Evolutionary Architecture & Mechanisms

### Central Pattern Generator (CPG)
A bio-inspired rhythmic oscillator that produces alternating sinusoidal control signals (sin, cos) to establish a base coordinated locomotion cycle for the quadruped legs.

### Adaptive CPG
A Central Pattern Generator whose phase advance rate is modulated by a learned NN output (5th motor output), allowing the creature to couple stride frequency to terrain slope automatically.

### Perlin Terrain
Procedurally generated continuous heightmap using 1D Perlin noise. Difficulty (amplitude + frequency) increases with x-displacement. Replaces hardcoded uphill/downhill segments.

### Tournament Selection (k=3)
GA parent selection where 3 random individuals compete; highest fitness wins. Replaces pure elite sampling to preserve population diversity.

### Terrain Lookahead
3 additional state features giving ground_y at +50, +100, +150px ahead of the torso. Grows state vector from 15 to 18 inputs. Basic exteroceptive terrain awareness without camera or LiDAR.

### Delta-State SSE
SSE frames transmitting only changed fields after the first full frame. Reduces stream bandwidth ~40-50% on flat terrain runs.
