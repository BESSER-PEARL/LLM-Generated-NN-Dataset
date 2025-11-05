'''
Generate a complete PyTorch NN code that satisfies the following four Requirements: 

Architecture: RNN-Simple 
Learning Task: Representation-learning
Input Type + Scale: Time series, seq_length: Small <50 and features: Small <50 
Complexity: Wide-Deep: Number of RNN-Simple layers at least 2, hidden_size of first RNN-Simple layer at least min(128, upper_bound_of_features).


- Provide complete PyTorch code for the model only.
- All layer parameters must be fixed values inside the network, with no variables or external constants used.
- Avoid creating excessively large layers that would cause unreasonable memory usage.
- The model must only use standard layers directly, without defining helper functions or custom block classes.
- Output only the code, with no comments.
'''
import torch
import torch.nn as nn

class RNNRepresentationModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.rnn1 = nn.RNN(input_size=32, hidden_size=64, num_layers=1, batch_first=True, nonlinearity='tanh', bias=True)
        self.rnn2 = nn.RNN(input_size=64, hidden_size=64, num_layers=1, batch_first=True, nonlinearity='tanh', bias=True)
        self.proj = nn.Linear(128, 64)
        self.norm = nn.LayerNorm(64)

    def forward(self, x):
        out1, h1 = self.rnn1(x)
        out2, h2 = self.rnn2(out1)
        h1_last = h1[-1]
        h2_last = h2[-1]
        z = torch.cat([h1_last, h2_last], dim=1)
        z = self.proj(z)
        z = self.norm(z)
        return z