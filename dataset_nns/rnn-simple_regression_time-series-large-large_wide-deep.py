'''
Generate a complete PyTorch NN code that satisfies the following four Requirements: 

Architecture: RNN-Simple 
Learning Task: Regression
Input Type + Scale: Time series, seq_length: Large >2000 and features: Large >2000 
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
        self.rnn1 = nn.RNN(input_size=2048, hidden_size=128, num_layers=1, batch_first=True, nonlinearity='tanh')
        self.rnn2 = nn.RNN(input_size=128, hidden_size=64, num_layers=1, batch_first=True, nonlinearity='tanh')
        self.dropout = nn.Dropout(p=0.2)
        self.wide = nn.Linear(2048, 32)
        self.relu = nn.ReLU()
        self.head1 = nn.Linear(96, 32)
        self.head2 = nn.Linear(32, 1)

    def forward(self, x):
        inp = x
        x, _ = self.rnn1(x)
        x, _ = self.rnn2(x)
        deep_feat = x[:, -1, :]
        deep_feat = self.dropout(deep_feat)
        wide_feat = self.relu(self.wide(inp[:, -1, :]))
        combined = torch.cat([deep_feat, wide_feat], dim=1)
        combined = self.relu(self.head1(combined))
        out = self.head2(combined)
        return out