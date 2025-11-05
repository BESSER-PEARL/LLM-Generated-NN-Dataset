'''
Generate a complete PyTorch NN code that satisfies the following four Requirements: 

Architecture: RNN-GRU 
Learning Task: Representation-learning
Input Type + Scale: Time series, seq_length: Small <50 and features: Large >2000 
Complexity: Simple: Number of RNN-GRU layers up to 2, hidden_size of first RNN-GRU layer up to min(128, upper_bound_of_features).


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
        self.gru1 = nn.GRU(input_size=2048, hidden_size=128, num_layers=1, batch_first=True, dropout=0.0, bidirectional=False)
        self.gru2 = nn.GRU(input_size=128, hidden_size=64, num_layers=1, batch_first=True, dropout=0.0, bidirectional=False)
        self.dropout = nn.Dropout(p=0.1)
        self.proj = nn.Linear(64, 64)
        self.norm = nn.LayerNorm(64)
        self.act = nn.Tanh()
    def forward(self, x):
        out1, _ = self.gru1(x)
        out1 = self.dropout(out1)
        out2, h2 = self.gru2(out1)
        rep = h2[-1]
        rep = self.proj(rep)
        rep = self.norm(rep)
        rep = self.act(rep)
        return rep