'''
Generate a complete PyTorch NN code that satisfies the following four Requirements: 

Architecture: RNN-LSTM 
Learning Task: Representation-learning
Input Type + Scale: Time series, seq_length: Small <50 and features: Large >2000 
Complexity: Deep: Number of RNN-LSTM layers at least 2.


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
        self.lstm = nn.LSTM(input_size=2048, hidden_size=256, num_layers=3, dropout=0.2, batch_first=True, bidirectional=False)
        self.proj = nn.Sequential(
            nn.Linear(256, 256, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(256, 128, bias=True),
            nn.LayerNorm(128, elementwise_affine=True)
        )

    def forward(self, x):
        _, (hn, _) = self.lstm(x)
        h = hn[-1]
        z = self.proj(h)
        return z