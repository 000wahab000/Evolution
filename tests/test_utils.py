import os

import numpy as np
from src.model import CreatureBrain
from src.utils import atomic_save, load_state_dict_file, load_universal_best_meta, save_universal_best


def test_atomic_save(tmp_path):
    save_path = str(tmp_path / "subdir" / "test_weights.npz")
    state_dict = {"w": np.array([1.0, 2.0, 3.0], dtype=np.float32)}

    atomic_save(state_dict, save_path)

    assert os.listdir(tmp_path / "subdir") == ["test_weights.npz"]  # no stray temp file
    loaded = load_state_dict_file(save_path)
    assert np.array_equal(state_dict["w"], loaded["w"])
    assert loaded["w"].dtype == np.float32


def test_brain_save_load_roundtrip(tmp_path):
    save_path = str(tmp_path / "brain.npz")
    brain = CreatureBrain()
    atomic_save(brain.state_dict(), save_path)

    restored = CreatureBrain()
    restored.load_state_dict(load_state_dict_file(save_path))
    state = np.random.randn(15).astype(np.float32)
    assert np.array_equal(brain.get_action(state), restored.get_action(state))


def test_save_and_load_universal_best(tmp_path):
    models_dir = str(tmp_path / "models")
    model = CreatureBrain()

    save_universal_best(
        model=model,
        fitness=123.45,
        generation=10,
        fitness_mode="baseline",
        pop_size=50,
        mutation_rate=0.15,
        models_dir=models_dir,
    )

    best_path = os.path.join(models_dir, "universal_best.npz")
    assert os.path.exists(best_path)

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
