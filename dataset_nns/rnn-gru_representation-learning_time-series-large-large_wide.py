'''
Generate a complete PyTorch NN code that satisfies the following four Requirements: 

Architecture: RNN-GRU 
Learning Task: Representation-learning
Input Type + Scale: Time series, seq_length: Large >2000 and features: Large >2000 
Complexity: Wide: hidden_size of first RNN-GRU layer at least min(128, upper_bound_of_features).


- Provide complete PyTorch code for the model only.
- All layer parameters must be fixed values inside the network, with no variables or external constants used.
- Avoid creating excessively large layers that would cause unreasonable memory usage.
- The model must only use standard layers directly, without defining helper functions or custom block classes.
- Output only the code, with no comments.
'''
import torch
import torch.nn as nn

class GRURepresentationModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.gru = nn.GRU(input_size=2048, hidden_size=128, num_layers=2, batch_first=True, bidirectional=True, dropout=0.1)
        self.proj = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
            nn.Linear(256, 128),
            nn.LayerNorm(128)
        )

    def forward(self, x):
        _, h = self.gru(x)
        rep = torch.cat([h[-2], h[-1]], dim=1)
        z = self.proj(rep)
        return z