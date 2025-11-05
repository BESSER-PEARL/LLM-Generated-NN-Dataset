'''
Generate a complete PyTorch NN code that satisfies the following four Requirements: 

Architecture: RNN-GRU 
Learning Task: Representation-learning
Input Type + Scale: Time series, seq_length: Small <50 and features: Large >2000 
Complexity: Wide: hidden_size of first RNN-GRU layer at least min(128, upper_bound_of_features).


- Provide complete PyTorch code for the model only.
- All layer parameters must be fixed values inside the network, with no variables or external constants used.
- Avoid creating excessively large layers that would cause unreasonable memory usage.
- The model must only use standard layers directly, without defining helper functions or custom block classes.
- Output only the code, with no comments.
'''
import torch
import torch.nn as nn

class GRURepresentationNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.gru1 = nn.GRU(input_size=2048, hidden_size=256, num_layers=1, batch_first=True)
        self.norm1 = nn.LayerNorm(256)
        self.dropout1 = nn.Dropout(0.2)
        self.gru2 = nn.GRU(input_size=256, hidden_size=128, num_layers=1, batch_first=True)
        self.norm2 = nn.LayerNorm(128)
        self.dropout2 = nn.Dropout(0.2)
        self.proj = nn.Linear(128, 64)

    def forward(self, x):
        out, _ = self.gru1(x)
        out = self.norm1(out)
        out = self.dropout1(out)
        out, _ = self.gru2(out)
        out = self.norm2(out)
        out = self.dropout2(out)
        rep = out[:, -1, :]
        z = self.proj(rep)
        return z