'''
Generate a complete PyTorch NN code that satisfies the following four Requirements: 

Architecture: RNN-Simple 
Learning Task: Regression
Input Type + Scale: Time series, seq_length: Small <50 and features: Small <50 
Complexity: Wide: hidden_size of first RNN-Simple layer at least min(128, upper_bound_of_features).


- Provide complete PyTorch code for the model only.
- All layer parameters must be fixed values inside the network, with no variables or external constants used.
- Avoid creating excessively large layers that would cause unreasonable memory usage.
- The model must only use standard layers directly, without defining helper functions or custom block classes.
- Output only the code, with no comments.
'''
import torch
import torch.nn as nn

class TimeSeriesRNNRegression(nn.Module):
    def __init__(self):
        super().__init__()
        self.rnn = nn.RNN(input_size=32, hidden_size=128, num_layers=1, batch_first=True, nonlinearity='tanh', bidirectional=False)
        self.fc1 = nn.Linear(128, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, 1)
    def forward(self, x):
        out, _ = self.rnn(x)
        x = out[:, -1, :]
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x