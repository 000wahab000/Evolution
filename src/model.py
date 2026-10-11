import numpy as np


class CreatureBrain:
    """15 -> 16 -> 4 tanh MLP. Weights keep nn.Linear's (out, in) layout."""

    def __init__(self, input_size=15, output_size=4):
        self.w1 = np.random.normal(0, 0.25, (16, input_size)).astype(np.float32)
        self.b1 = np.random.normal(0, 0.25, 16).astype(np.float32)
        self.w2 = np.random.normal(0, 0.25, (output_size, 16)).astype(np.float32)
        self.b2 = np.random.normal(0, 0.25, output_size).astype(np.float32)

    def get_action(self, state):
        x = np.asarray(state, dtype=np.float32)
        return np.tanh(self.w2 @ np.tanh(self.w1 @ x + self.b1) + self.b2)

    def parameters(self):
        return [self.w1, self.b1, self.w2, self.b2]

    def state_dict(self):
        return {"w1": self.w1, "b1": self.b1, "w2": self.w2, "b2": self.b2}

    def load_state_dict(self, state_dict):
        for key in ("w1", "b1", "w2", "b2"):
            setattr(self, key, np.array(state_dict[key], dtype=np.float32))
