'''
Generate a complete PyTorch NN code that satisfies the following four Requirements: 

Architecture: RNN-Simple 
Learning Task: Regression
Input Type + Scale: Time series, seq_length: Small <50 and features: Small <50 
Complexity: Wide-Deep: Number of RNN-Simple layers at least 2, hidden_size of first RNN-Simple layer at least min(128, upper_bound_of_features).


- Provide complete PyTorch code for the model only.
- All layer parameters must be fixed values inside the network, with no variables or external constants used.
- Avoid creating excessively large layers that would cause unreasonable memory usage.
- The model must only use standard layers directly, without defining helper functions or custom block classes.
- Output only the code, with no comments.
'''
import torch
import torch.nn as nn

class RNNRegression(nn.Module):
    def __init__(self):
        super().__init__()
        self.rnn1 = nn.RNN(input_size=32, hidden_size=64, num_layers=1, nonlinearity='tanh', batch_first=True, bias=True, dropout=0.0, bidirectional=False)
        self.rnn2 = nn.RNN(input_size=64, hidden_size=64, num_layers=1, nonlinearity='tanh', batch_first=True, bias=True, dropout=0.0, bidirectional=False)
        self.fc1 = nn.Linear(64, 32)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(32, 1)

    def forward(self, x):
        out, _ = self.rnn1(x)
        out, _ = self.rnn2(out)
        out = out[:, -1, :]
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        return out