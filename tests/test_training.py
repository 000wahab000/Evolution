import os
from main import train


def test_train_integration(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    train(generations=1, pop_size=2, fitness_mode="baseline")

    assert os.path.exists(os.path.join("models", "best_model.npz"))
    assert os.path.exists(os.path.join("models", "universal_best.npz"))
    assert os.path.exists(os.path.join("models", "universal_best_meta.json"))
    assert os.path.exists(os.path.join("logs", "history.csv"))
