import torch
import torch.nn as nn

class CreatureBrain(nn.Module):
    def __init__(self, input_size=13, output_size=4):
        super(CreatureBrain, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 16),
            nn.Tanh(),
            nn.Linear(16, output_size),
            nn.Tanh() # Output actions between -1 and 1
        )
        # Initialize weights with standard normal for GA mutation baseline
        for param in self.net.parameters():
            nn.init.normal_(param, 0, 1)

    def forward(self, x):
        return self.net(x)

    def get_action(self, state):
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state)
            action = self.forward(state_tensor)
            return action.numpy()
