import torch
import torch.nn as nn


class CreatureBrain(nn.Module):
    def __init__(self, input_size=15, output_size=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 16),
            nn.Tanh(),
            nn.Linear(16, output_size),
            nn.Tanh(),
        )
        for param in self.net.parameters():
            nn.init.normal_(param, 0, 0.25)

    def forward(self, x):
        return self.net(x)

    def get_action(self, state):
        with torch.no_grad():
            state_tensor = torch.as_tensor(state, dtype=torch.float32)
            return self.net(state_tensor).numpy()
