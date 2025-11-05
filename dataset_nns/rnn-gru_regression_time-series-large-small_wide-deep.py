'''
Generate a complete PyTorch NN code that satisfies the following four Requirements: 

Architecture: RNN-GRU 
Learning Task: Regression
Input Type + Scale: Time series, seq_length: Large >2000 and features: Small <50 
Complexity: Wide-Deep: Number of RNN-GRU layers at least 2, hidden_size of first RNN-GRU layer at least min(128, upper_bound_of_features).


- Provide complete PyTorch code for the model only.
- All layer parameters must be fixed values inside the network, with no variables or external constants used.
- Avoid creating excessively large layers that would cause unreasonable memory usage.
- The model must only use standard layers directly, without defining helper functions or custom block classes.
- Output only the code, with no comments.
'''
import torch
import torch.nn as nn

class GRUWideDeepRegressor(nn.Module):
    def __init__(self):
        super().__init__()
        self.gru1 = nn.GRU(input_size=32, hidden_size=64, num_layers=1, batch_first=True)
        self.gru2 = nn.GRU(input_size=64, hidden_size=32, num_layers=1, batch_first=True)
        self.wide = nn.Linear(32, 16)
        self.act = nn.ReLU()
        self.fc1 = nn.Linear(48, 64)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x):
        out1, _ = self.gru1(x)
        out2, _ = self.gru2(out1)
        last = out2[:, -1, :]
        b, t, _ = x.shape
        w = self.wide(x.reshape(b * t, 32))
        w = self.act(w)
        w = w.reshape(b, t, 16).mean(dim=1)
        h = torch.cat([last, w], dim=-1)
        h = self.act(self.fc1(h))
        y = self.fc2(h)
        return y