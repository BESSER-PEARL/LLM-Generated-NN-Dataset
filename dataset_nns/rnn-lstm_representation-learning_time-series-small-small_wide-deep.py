'''
Generate a complete PyTorch NN code that satisfies the following four Requirements: 

Architecture: RNN-LSTM 
Learning Task: Representation-learning
Input Type + Scale: Time series, seq_length: Small <50 and features: Small <50 
Complexity: Wide-Deep: Number of RNN-LSTM layers at least 2, hidden_size of first RNN-LSTM layer at least min(128, upper_bound_of_features).


- Provide complete PyTorch code for the model only.
- All layer parameters must be fixed values inside the network, with no variables or external constants used.
- Avoid creating excessively large layers that would cause unreasonable memory usage.
- The model must only use standard layers directly, without defining helper functions or custom block classes.
- Output only the code, with no comments.
'''
import torch
import torch.nn as nn

class LSTMRepresentationModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm1 = nn.LSTM(input_size=32, hidden_size=64, num_layers=1, batch_first=True, bidirectional=True)
        self.dropout1 = nn.Dropout(p=0.2)
        self.lstm2 = nn.LSTM(input_size=128, hidden_size=64, num_layers=1, batch_first=True, bidirectional=True)
        self.dropout2 = nn.Dropout(p=0.2)
        self.layernorm = nn.LayerNorm(128)
        self.proj1 = nn.Linear(128, 96)
        self.activation = nn.ReLU()
        self.proj2 = nn.Linear(96, 64)

    def forward(self, x):
        y, _ = self.lstm1(x)
        y = self.dropout1(y)
        y, _ = self.lstm2(y)
        y = self.dropout2(y)
        y = y.mean(dim=1)
        y = self.layernorm(y)
        y = self.proj1(y)
        y = self.activation(y)
        y = self.proj2(y)
        return y