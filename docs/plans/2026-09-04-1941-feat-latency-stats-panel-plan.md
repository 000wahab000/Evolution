---
artifact_contract: ce-unified-plan/v1
artifact_readiness: ready-for-work
product_contract_source: ce-brainstorm
plan_type: feat
date: 2026-09-04
---

# Goal Capsule

Add a real-time latency telemetry stats panel in `main.py --render` to measure and display per-stage execution times (NN inference, physics simulation, Pymunk debug drawing, UI/stats, display flip), detect and highlight the primary bottleneck, and allow real-time diagnostic evaluation of terrain rendering costs.

# Product Contract

## Requirements
1. **Per-Stage Latency Profiling**: Frame-by-frame measurement of:
   - Neural Net Inference (`models.get_action`)
   - Physics Step (`envs.step`)
   - Pymunk Render (`space.debug_draw`)
   - UI / Overlay draw
   - Display Flip / Vsync wait (`pygame.display.flip`, `clock.tick`)
2. **Visual HUD Panel**:
   - Placed on the top-right of the Pygame screen (`x=520, y=20`) to complement the leader stats on top-left.
   - Semi-transparent dark background card (`(20, 20, 25, 210)`) for crisp readability.
   - Real-time FPS and total frame time with color coding (Green/Yellow/Red).
3. **Bottleneck Callout**:
   - Automatically identify whichever component has the highest latency and display `[BOTTLENECK]: <Stage> (<percent>%)` in high-visibility red.
4. **EMA Smoothing**:
   - Exponential Moving Average with $\alpha = 0.08$ to ensure steady readability without jitter.
5. **Ponytail Compliance**:
   - Use `time.perf_counter()` from Python stdlib.
   - Zero added external dependencies.
   - Minimal lines of code, zero heap churn in the render loop.

# Technical Specification

- **File**: `main.py`
- **Location**: Function `render()`
- **Timers**:
  ```python
  t0 = time.perf_counter()
  # NN
  t1 = time.perf_counter()
  # Physics
  t2 = time.perf_counter()
  # Render
  t3 = time.perf_counter()
  # UI
  t4 = time.perf_counter()
  # Flip/Tick
  t5 = time.perf_counter()
  ```
- **Smoothing formula**:
  `ema_val = ema_val * (1 - alpha) + current_val * alpha`

# Verification
- Run `pytest` to ensure all existing environment tests pass.
- Run `python main.py --render` to verify HUD visibility, live numbers, and bottleneck highlight.
