'''
Generate a complete PyTorch NN code that satisfies the following four Requirements: 

Architecture: RNN-Simple 
Learning Task: Representation-learning
Input Type + Scale: Time series, seq_length: Large >2000 and features: Small <50 
Complexity: Wide-Deep: Number of RNN-Simple layers at least 2, hidden_size of first RNN-Simple layer at least min(128, upper_bound_of_features).


- Provide complete PyTorch code for the model only.
- All layer parameters must be fixed values inside the network, with no variables or external constants used.
- Avoid creating excessively large layers that would cause unreasonable memory usage.
- The model must only use standard layers directly, without defining helper functions or custom block classes.
- Output only the code, with no comments.
'''
import torch
from torch import nn

class RNNRepresentation(nn.Module):
    def __init__(self):
        super().__init__()
        self.rnn1 = nn.RNN(input_size=32, hidden_size=64, num_layers=1, batch_first=True, nonlinearity='tanh')
        self.rnn2 = nn.RNN(input_size=64, hidden_size=64, num_layers=1, batch_first=True, nonlinearity='tanh')
        self.dropout = nn.Dropout(p=0.1)
        self.proj = nn.Linear(64, 128)
        self.norm = nn.LayerNorm(128)
        self.activation = nn.Tanh()
    def forward(self, x):
        out, _ = self.rnn1(x)
        out = self.dropout(out)
        out, _ = self.rnn2(out)
        rep = out.mean(dim=1)
        rep = self.proj(rep)
        rep = self.norm(rep)
        rep = self.activation(rep)
        return rep