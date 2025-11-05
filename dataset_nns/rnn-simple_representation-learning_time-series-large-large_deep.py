'''
Generate a complete PyTorch NN code that satisfies the following four Requirements: 

Architecture: RNN-Simple 
Learning Task: Representation-learning
Input Type + Scale: Time series, seq_length: Large >2000 and features: Large >2000 
Complexity: Deep: Number of RNN-Simple layers at least 2.


- Provide complete PyTorch code for the model only.
- All layer parameters must be fixed values inside the network, with no variables or external constants used.
- Avoid creating excessively large layers that would cause unreasonable memory usage.
- The model must only use standard layers directly, without defining helper functions or custom block classes.
- Output only the code, with no comments.
'''
import torch
import torch.nn as nn

class RNNRepresentation(nn.Module):
    def __init__(self):
        super().__init__()
        self.rnn = nn.RNN(input_size=2048, hidden_size=256, num_layers=3, nonlinearity='tanh', batch_first=True, dropout=0.1, bidirectional=False)
        self.proj = nn.Linear(256, 128)
        self.act = nn.Tanh()
    def forward(self, x):
        _, h_n = self.rnn(x)
        h = h_n[-1]
        z = self.proj(h)
        z = self.act(z)
        return z