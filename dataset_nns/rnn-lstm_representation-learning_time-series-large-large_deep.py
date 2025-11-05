'''
Generate a complete PyTorch NN code that satisfies the following four Requirements: 

Architecture: RNN-LSTM 
Learning Task: Representation-learning
Input Type + Scale: Time series, seq_length: Large >2000 and features: Large >2000 
Complexity: Deep: Number of RNN-LSTM layers at least 2.


- Provide complete PyTorch code for the model only.
- All layer parameters must be fixed values inside the network, with no variables or external constants used.
- Avoid creating excessively large layers that would cause unreasonable memory usage.
- The model must only use standard layers directly, without defining helper functions or custom block classes.
- Output only the code, with no comments.
'''
import torch
import torch.nn as nn

class TimeSeriesRepresentationModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(input_size=2048, hidden_size=64, num_layers=3, batch_first=True, dropout=0.1, bidirectional=False)
        self.norm = nn.LayerNorm(64)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.proj = nn.Sequential(
            nn.Linear(64, 128, bias=True),
            nn.ReLU(inplace=False),
            nn.Dropout(p=0.1),
            nn.Linear(128, 128, bias=True)
        )

    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.norm(x)
        x = x.transpose(1, 2)
        x = self.pool(x).squeeze(-1)
        x = self.proj(x)
        return x