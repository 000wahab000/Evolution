# Project Agent Guidelines & Conventions

## Workspace Directory Structure

- **`src/`**: Core python modules (`environment.py`, `evolution.py`, `model.py`).
- **`models/`**: Saved PyTorch model checkpoints (`best_model.pth`, `top_0.pth` ... `top_4.pth`).
- **`logs/`**: Training logs and metrics (`history.csv`, `evolution.log`).
- **`templates/`**: Web application HTML templates (`index.html`).
- **`tests/`**: Pytest test suite (`test_environment.py`).
- **`assets/`**: Media assets (GIFs, screenshots, diagrams).

## Model & File Location Rules

1. **Model Checkpoints**: Always save and load model checkpoints inside the `models/` directory (e.g. `models/best_model.pth`, `models/top_X.pth`).
2. **Logs & Metrics**: Always output training logs, metrics CSVs, and progress reports inside `logs/`.
3. **No Root Pollution**: Do not leave transient `.pth` or `.log` files sitting in the repository root directory.

## Ponytail / Senior Developer Principles

1. **YAGNI & Simplicity**: Minimal code that works. Avoid unnecessary abstractions, boilerplate, or unrequested dependencies.
2. **Root Cause Fixes**: Fix bugs at the root cause, not symptom-swallowing try/except blocks.
3. **Clean Codebase**: Keep the repository root uncluttered and organized.
