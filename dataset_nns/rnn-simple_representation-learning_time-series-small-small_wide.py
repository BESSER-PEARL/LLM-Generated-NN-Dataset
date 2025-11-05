'''
Generate a complete PyTorch NN code that satisfies the following four Requirements: 

Architecture: RNN-Simple 
Learning Task: Representation-learning
Input Type + Scale: Time series, seq_length: Small <50 and features: Small <50 
Complexity: Wide: hidden_size of first RNN-Simple layer at least min(128, upper_bound_of_features).


- Provide complete PyTorch code for the model only.
- All layer parameters must be fixed values inside the network, with no variables or external constants used.
- Avoid creating excessively large layers that would cause unreasonable memory usage.
- The model must only use standard layers directly, without defining helper functions or custom block classes.
- Output only the code, with no comments.
'''
import torch
import torch.nn as nn

class SimpleRNNRepresentation(nn.Module):
    def __init__(self):
        super().__init__()
        self.rnn = nn.RNN(input_size=32, hidden_size=64, num_layers=1, batch_first=True, nonlinearity='tanh', bias=True)
        self.dropout = nn.Dropout(p=0.1)
        self.proj = nn.Linear(64, 32)
        self.act = nn.Tanh()
    def forward(self, x):
        out, _ = self.rnn(x)
        h = out[:, -1, :]
        z = self.dropout(h)
        z = self.proj(z)
        z = self.act(z)
        return z