'''
Generate a complete PyTorch NN code that satisfies the following four Requirements: 

Architecture: RNN-GRU 
Learning Task: Regression
Input Type + Scale: Time series, seq_length: Large >2000 and features: Large >2000 
Complexity: Deep: Number of RNN-GRU layers at least 2.


- Provide complete PyTorch code for the model only.
- All layer parameters must be fixed values inside the network, with no variables or external constants used.
- Avoid creating excessively large layers that would cause unreasonable memory usage.
- The model must only use standard layers directly, without defining helper functions or custom block classes.
- Output only the code, with no comments.
'''
import torch
from torch import nn

class GRURegression(nn.Module):
    def __init__(self):
        super().__init__()
        self.gru = nn.GRU(input_size=2048, hidden_size=128, num_layers=2, dropout=0.2, batch_first=True)
        self.dropout = nn.Dropout(p=0.3)
        self.fc1 = nn.Linear(128, 64)
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x):
        _, h = self.gru(x)
        y = h[-1]
        y = self.dropout(y)
        y = self.fc1(y)
        y = self.act(y)
        y = self.dropout(y)
        y = self.fc2(y)
        return y