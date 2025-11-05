'''
Generate a complete PyTorch NN code that satisfies the following four Requirements: 

Architecture: RNN-GRU 
Learning Task: Representation-learning
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
import torch.nn.functional as F

class RNNGRURepresentationModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.gru1 = nn.GRU(input_size=32, hidden_size=64, num_layers=1, batch_first=True, bidirectional=True)
        self.gru2 = nn.GRU(input_size=128, hidden_size=64, num_layers=1, batch_first=True, bidirectional=True)
        self.ln = nn.LayerNorm(448)
        self.drop = nn.Dropout(p=0.1)
        self.fc1 = nn.Linear(448, 256)
        self.fc2 = nn.Linear(256, 128)

    def forward(self, x):
        y, _ = self.gru1(x)
        y, _ = self.gru2(y)
        deep_mean = y.mean(dim=1)
        deep_max, _ = y.max(dim=1)
        deep_last = y[:, -1, :]
        deep = torch.cat([deep_mean, deep_max, deep_last], dim=-1)
        wide_mean = x.mean(dim=1)
        wide_max, _ = x.max(dim=1)
        wide = torch.cat([wide_mean, wide_max], dim=-1)
        h = torch.cat([deep, wide], dim=-1)
        h = self.ln(h)
        h = self.drop(h)
        h = self.fc1(h)
        h = F.gelu(h)
        h = self.drop(h)
        z = self.fc2(h)
        return z