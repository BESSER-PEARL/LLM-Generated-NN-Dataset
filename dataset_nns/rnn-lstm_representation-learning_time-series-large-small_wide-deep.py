'''
Generate a complete PyTorch NN code that satisfies the following four Requirements: 

Architecture: RNN-LSTM 
Learning Task: Representation-learning
Input Type + Scale: Time series, seq_length: Large >2000 and features: Small <50 
Complexity: Wide-Deep: Number of RNN-LSTM layers at least 2, hidden_size of first RNN-LSTM layer at least min(128, upper_bound_of_features).


- Provide complete PyTorch code for the model only.
- All layer parameters must be fixed values inside the network, with no variables or external constants used.
- Avoid creating excessively large layers that would cause unreasonable memory usage.
- The model must only use standard layers directly, without defining helper functions or custom block classes.
- Output only the code, with no comments.
'''
import torch
import torch.nn as nn

class RNNLSTMRepresentation(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm1 = nn.LSTM(input_size=48, hidden_size=128, num_layers=1, batch_first=True, dropout=0.0, bidirectional=False)
        self.lstm2 = nn.LSTM(input_size=128, hidden_size=128, num_layers=1, batch_first=True, dropout=0.0, bidirectional=False)
        self.norm = nn.LayerNorm(128)
        self.proj = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.LayerNorm(64)
        )
    def forward(self, x):
        y, _ = self.lstm1(x)
        y, _ = self.lstm2(y)
        y = self.norm(y.mean(dim=1))
        z = self.proj(y)
        return z