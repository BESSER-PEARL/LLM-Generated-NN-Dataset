'''
Generate a complete PyTorch NN code that satisfies the following four Requirements: 

Architecture: RNN-Simple 
Learning Task: Representation-learning
Input Type + Scale: Time series, seq_length: Large >2000 and features: Small <50 
Complexity: Deep: Number of RNN-Simple layers at least 2.


- Provide complete PyTorch code for the model only.
- All layer parameters must be fixed values inside the network, with no variables or external constants used.
- Avoid creating excessively large layers that would cause unreasonable memory usage.
- The model must only use standard layers directly, without defining helper functions or custom block classes.
- Output only the code, with no comments.
'''
import torch
import torch.nn as nn

class TimeSeriesRNNRepresentation(nn.Module):
    def __init__(self):
        super().__init__()
        self.rnn1 = nn.RNN(input_size=32, hidden_size=128, num_layers=1, batch_first=True, nonlinearity='tanh')
        self.rnn2 = nn.RNN(input_size=128, hidden_size=128, num_layers=1, batch_first=True, nonlinearity='tanh')
        self.proj = nn.Linear(128, 64)
        self.norm = nn.LayerNorm(64)
        self.act = nn.Tanh()

    def forward(self, x):
        y1, _ = self.rnn1(x)
        y2, h2 = self.rnn2(y1)
        z = self.proj(h2[-1])
        z = self.norm(z)
        z = self.act(z)
        return z