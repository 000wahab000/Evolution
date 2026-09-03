import os
import pytest
import torch
import torch.nn as nn
from src.utils import atomic_torch_save, load_universal_best_meta, save_universal_best


class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4, 2)


def test_atomic_torch_save(tmp_path):
    save_path = str(tmp_path / "subdir" / "test_tensor.pth")
    tensor_data = torch.tensor([1.0, 2.0, 3.0])

    atomic_torch_save(tensor_data, save_path)

    assert os.path.exists(save_path)
    loaded = torch.load(save_path, weights_only=True)
    assert torch.equal(tensor_data, loaded)


def test_save_and_load_universal_best(tmp_path):
    models_dir = str(tmp_path / "models")
    model = SimpleModel()

    save_universal_best(
        model=model,
        fitness=123.45,
        generation=10,
        fitness_mode="baseline",
        pop_size=50,
        mutation_rate=0.15,
        models_dir=models_dir,
    )

    best_pth = os.path.join(models_dir, "universal_best.pth")
    assert os.path.exists(best_pth)

    meta = load_universal_best_meta(models_dir)
    assert meta["best_fitness"] == 123.45
    assert meta["generation"] == 10
    assert meta["fitness_mode"] == "baseline"
    assert meta["pop_size"] == 50
    assert meta["mutation_rate"] == 0.15
    assert "saved_at" in meta


def test_load_universal_best_meta_empty_or_missing(tmp_path):
    # Non-existent directory
    missing_dir = str(tmp_path / "non_existent")
    assert load_universal_best_meta(missing_dir) == {}

    # Corrupted file
    corrupt_dir = str(tmp_path / "corrupt")
    os.makedirs(corrupt_dir, exist_ok=True)
    with open(os.path.join(corrupt_dir, "universal_best_meta.json"), "w") as f:
        f.write("{invalid_json")
    assert load_universal_best_meta(corrupt_dir) == {}
