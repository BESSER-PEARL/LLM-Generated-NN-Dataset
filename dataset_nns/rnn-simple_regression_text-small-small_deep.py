'''
Generate a complete PyTorch NN code that satisfies the following four Requirements: 

Architecture: RNN-Simple 
Learning Task: Regression
Input Type + Scale: Text, seq_length: Small <50 and vocab_size: Small <1k 
Complexity: Deep: Number of RNN-Simple layers at least 2.


- Provide complete PyTorch code for the model only.
- All layer parameters must be fixed values inside the network, with no variables or external constants used.
- Avoid creating excessively large layers that would cause unreasonable memory usage.
- The model must only use standard layers directly, without defining helper functions or custom block classes.
- Output only the code, with no comments.
'''
import torch
import torch.nn as nn

class RNNTextRegressor(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(500, 32)
        self.rnn = nn.RNN(input_size=32, hidden_size=64, num_layers=2, nonlinearity='tanh', batch_first=True)
        self.fc = nn.Linear(64, 1)
    def forward(self, x):
        x = self.embedding(x)
        out, h_n = self.rnn(x)
        h_last = h_n[-1]
        y = self.fc(h_last)
        return y