'''
Generate a complete PyTorch NN code that satisfies the following four Requirements: 

Architecture: RNN-Simple 
Learning Task: Regression
Input Type + Scale: Text, seq_length: Large >2000 and vocab_size: Small <1k 
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
        self.embedding = nn.Embedding(512, 64)
        self.rnn = nn.RNN(64, 96, num_layers=3, nonlinearity='tanh', batch_first=True, dropout=0.2, bidirectional=False)
        self.fc1 = nn.Linear(96, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, 1)
    def forward(self, x):
        x = self.embedding(x)
        out, _ = self.rnn(x)
        last = out[:, -1, :]
        y = self.fc1(last)
        y = self.relu(y)
        y = self.fc2(y)
        return y