'''
Generate a complete PyTorch NN code that satisfies the following four Requirements: 

Architecture: RNN-Simple 
Learning Task: Regression
Input Type + Scale: Time series, seq_length: Large >2000 and features: Small <50 
Complexity: Wide-Deep: Number of RNN-Simple layers at least 2, hidden_size of first RNN-Simple layer at least min(128, upper_bound_of_features).


- Provide complete PyTorch code for the model only.
- All layer parameters must be fixed values inside the network, with no variables or external constants used.
- Avoid creating excessively large layers that would cause unreasonable memory usage.
- The model must only use standard layers directly, without defining helper functions or custom block classes.
- Output only the code, with no comments.
'''
import torch
import torch.nn as nn

class RNNRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.rnn = nn.RNN(input_size=32, hidden_size=64, num_layers=2, batch_first=True, nonlinearity='tanh', dropout=0.0, bidirectional=False)
        self.fc = nn.Linear(64, 1)
    def forward(self, x):
        out, _ = self.rnn(x)
        out = out[:, -1, :]
        y = self.fc(out)
        return y