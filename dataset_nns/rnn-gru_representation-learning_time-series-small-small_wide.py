'''
Generate a complete PyTorch NN code that satisfies the following four Requirements: 

Architecture: RNN-GRU 
Learning Task: Representation-learning
Input Type + Scale: Time series, seq_length: Small <50 and features: Small <50 
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
        self.gru1 = nn.GRU(input_size=32, hidden_size=64, num_layers=1, batch_first=True, bidirectional=True)
        self.gru2 = nn.GRU(input_size=128, hidden_size=64, num_layers=1, batch_first=True, bidirectional=True)
        self.projection = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
            nn.Linear(128, 128),
            nn.LayerNorm(128)
        )

    def forward(self, x):
        x, _ = self.gru1(x)
        x, _ = self.gru2(x)
        x = x.mean(dim=1)
        x = self.projection(x)
        return x