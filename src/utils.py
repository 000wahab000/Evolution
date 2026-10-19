import json
import os
import time

import numpy as np

MODELS_DIR = os.path.join("data", "models")
LOGS_DIR = os.path.join("data", "logs")
UNIVERSAL_BEST_PATH = os.path.join(MODELS_DIR, "universal_best.npz")
UNIVERSAL_META_PATH = os.path.join(MODELS_DIR, "universal_best_meta.json")


def atomic_save(state_dict: dict, path: str) -> None:
    """Save a state_dict of NumPy arrays (.npz) atomically via temp file replacement."""
    dir_name = os.path.dirname(path)
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)
    tmp_path = path + ".tmp.npz"  # np.savez appends ".npz" to names that lack it
    np.savez(tmp_path, **state_dict)
    os.replace(tmp_path, path)


def load_state_dict_file(path: str) -> dict:
    """Load a state_dict saved by atomic_save."""
    with np.load(path) as data:
        return {k: data[k] for k in data.files}


def load_universal_best_meta(models_dir: str = MODELS_DIR) -> dict:
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
    model,
    fitness: float,
    generation: int,
    fitness_mode: str,
    pop_size: int,
    mutation_rate: float,
    models_dir: str = MODELS_DIR,
) -> None:
    """Atomically persist the universal best model weights and metadata."""
    os.makedirs(models_dir, exist_ok=True)
    best_path = os.path.join(models_dir, "universal_best.npz")
    meta_path = os.path.join(models_dir, "universal_best_meta.json")

    atomic_save(model.state_dict(), best_path)
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
