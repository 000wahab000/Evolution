import json
import os
import time

import torch

DEFAULT_MODELS_DIR = "models"
UNIVERSAL_BEST_PATH = os.path.join(DEFAULT_MODELS_DIR, "universal_best.pth")
UNIVERSAL_META_PATH = os.path.join(DEFAULT_MODELS_DIR, "universal_best_meta.json")


def atomic_torch_save(obj, path: str) -> None:
    """Save PyTorch tensor or state_dict atomically via temp file replacement."""
    dir_name = os.path.dirname(path)
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)
    tmp_path = path + ".tmp"
    torch.save(obj, tmp_path)
    os.replace(tmp_path, path)


def load_universal_best_meta(models_dir: str = DEFAULT_MODELS_DIR) -> dict:
    """Load JSON metadata associated with the all-time best model checkpoint."""
    meta_path = os.path.join(models_dir, "universal_best_meta.json")
    if os.path.exists(meta_path):
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def save_universal_best(
    model: torch.nn.Module,
    fitness: float,
    generation: int,
    fitness_mode: str,
    pop_size: int,
    mutation_rate: float,
    models_dir: str = DEFAULT_MODELS_DIR,
) -> None:
    """Atomically persist the universal best model weights and metadata."""
    os.makedirs(models_dir, exist_ok=True)
    best_path = os.path.join(models_dir, "universal_best.pth")
    meta_path = os.path.join(models_dir, "universal_best_meta.json")

    atomic_torch_save(model.state_dict(), best_path)
    meta = {
        "best_fitness": float(fitness),
        "generation": int(generation),
        "fitness_mode": str(fitness_mode),
        "pop_size": int(pop_size),
        "mutation_rate": float(mutation_rate),
        "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    tmp_meta = meta_path + ".tmp"
    with open(tmp_meta, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    os.replace(tmp_meta, meta_path)
