import numpy as np
import torch
from src.model import CreatureBrain


def test_brain_initialization():
    brain = CreatureBrain(input_size=13, output_size=4)
    assert brain is not None
    param_count = sum(p.numel() for p in brain.parameters())
    assert param_count == (13 * 16 + 16) + (16 * 4 + 4)  # 224 + 68 = 292


def test_brain_action_range_and_shape():
    brain = CreatureBrain()
    state = np.random.randn(13).astype(np.float32)
    action = brain.get_action(state)

    assert isinstance(action, np.ndarray)
    assert action.shape == (4,)
    assert np.all(action >= -1.0) and np.all(action <= 1.0)
