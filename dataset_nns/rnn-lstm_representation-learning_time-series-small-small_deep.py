'''
Generate a complete PyTorch NN code that satisfies the following four Requirements: 

Architecture: RNN-LSTM 
Learning Task: Representation-learning
Input Type + Scale: Time series, seq_length: Small <50 and features: Small <50 
Complexity: Deep: Number of RNN-LSTM layers at least 2.


- Provide complete PyTorch code for the model only.
- All layer parameters must be fixed values inside the network, with no variables or external constants used.
- Avoid creating excessively large layers that would cause unreasonable memory usage.
- The model must only use standard layers directly, without defining helper functions or custom block classes.
- Output only the code, with no comments.
'''
import torch
import torch.nn as nn

class DeepLSTMRepresentation(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(input_size=16, hidden_size=64, num_layers=3, batch_first=True, dropout=0.1, bidirectional=True)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(0.1)
        self.proj1 = nn.Linear(128, 64)
        self.act = nn.ReLU()
        self.proj2 = nn.Linear(64, 32)

    def forward(self, x):
        y, _ = self.lstm(x)
        y = y.transpose(1, 2)
        p = self.pool(y).squeeze(-1)
        z = self.dropout(p)
        z = self.proj1(z)
        z = self.act(z)
        z = self.dropout(z)
        z = self.proj2(z)
        return z