'''
Generate a complete PyTorch NN code that satisfies the following four Requirements: 

Architecture: RNN-LSTM 
Learning Task: Representation-learning
Input Type + Scale: Time series, seq_length: Small <50 and features: Small <50 
Complexity: Wide: hidden_size of first RNN-LSTM layer at least min(128, upper_bound_of_features).


- Provide complete PyTorch code for the model only.
- All layer parameters must be fixed values inside the network, with no variables or external constants used.
- Avoid creating excessively large layers that would cause unreasonable memory usage.
- The model must only use standard layers directly, without defining helper functions or custom block classes.
- Output only the code, with no comments.
'''
import torch
import torch.nn as nn

class RNNLSTMEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm1 = nn.LSTM(input_size=32, hidden_size=128, num_layers=1, batch_first=True, bidirectional=False)
        self.dropout1 = nn.Dropout(p=0.2)
        self.lstm2 = nn.LSTM(input_size=128, hidden_size=64, num_layers=1, batch_first=True, bidirectional=False)
        self.dropout2 = nn.Dropout(p=0.2)
        self.proj = nn.Linear(in_features=128, out_features=64, bias=True)
        self.norm = nn.LayerNorm(normalized_shape=64)
        self.act = nn.ReLU()

    def forward(self, x):
        y, _ = self.lstm1(x)
        y = self.dropout1(y)
        y2, _ = self.lstm2(y)
        y2 = self.dropout2(y2)
        last = y2[:, -1, :]
        mean = y2.mean(dim=1)
        z = torch.cat([last, mean], dim=1)
        z = self.proj(z)
        z = self.norm(z)
        z = self.act(z)
        return z