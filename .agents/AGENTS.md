# Project Agent Guidelines & Conventions

## Workspace Directory Structure

- **`src/`**: Core python modules (`environment.py`, `evolution.py`, `model.py`).
- **`data/models/`**: Saved NumPy model checkpoints (`best_model.npz`, `top_0.npz` ... `top_4.npz`).
- **`data/logs/`**: Training logs and metrics (`history.csv`, `evolution.log`).
- **`web/`**: Dashboard template (`web/templates/index.html`) and recorded replays (`web/static/replays/`).
- **`tests/`**: Pytest test suite (`test_environment.py`).
- **`docs/`**: Study guide, `STRATEGY.md`, plans, and README media (`docs/media/`).
- **`docs/solutions/`**: Documented solutions to past technical challenges and bug fixes.

## Model & File Location Rules

1. **Model Checkpoints**: Always save and load model checkpoints inside the `data/models/` directory (e.g. `data/models/best_model.npz`, `data/models/top_X.npz`).
2. **Logs & Metrics**: Always output training logs, metrics CSVs, and progress reports inside `data/logs/`.
3. **No Root Pollution**: Do not leave transient `.npz` or `.log` files sitting in the repository root directory.

## Ponytail / Senior Developer Principles

1. **YAGNI & Simplicity**: Minimal code that works. Avoid unnecessary abstractions, boilerplate, or unrequested dependencies.
2. **Root Cause Fixes**: Fix bugs at the root cause, not symptom-swallowing try/except blocks.
3. **Clean Codebase**: Keep the repository root uncluttered and organized.
