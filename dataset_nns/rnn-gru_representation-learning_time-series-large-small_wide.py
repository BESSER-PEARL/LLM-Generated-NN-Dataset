'''
Generate a complete PyTorch NN code that satisfies the following four Requirements: 

Architecture: RNN-GRU 
Learning Task: Representation-learning
Input Type + Scale: Time series, seq_length: Large >2000 and features: Small <50 
Complexity: Wide: hidden_size of first RNN-GRU layer at least min(128, upper_bound_of_features).


- Provide complete PyTorch code for the model only.
- All layer parameters must be fixed values inside the network, with no variables or external constants used.
- Avoid creating excessively large layers that would cause unreasonable memory usage.
- The model must only use standard layers directly, without defining helper functions or custom block classes.
- Output only the code, with no comments.
'''
import torch
import torch.nn as nn

class GRURepresentation(nn.Module):
    def __init__(self):
        super().__init__()
        self.gru = nn.GRU(input_size=32, hidden_size=64, num_layers=2, batch_first=True, dropout=0.1, bidirectional=False)
        self.norm = nn.LayerNorm(64)
        self.proj1 = nn.Linear(64, 128)
        self.act = nn.ReLU()
        self.proj2 = nn.Linear(128, 64)

    def forward(self, x):
        out, _ = self.gru(x)
        out = self.norm(out)
        pooled = out.mean(dim=1)
        z = self.proj2(self.act(self.proj1(pooled)))
        return z