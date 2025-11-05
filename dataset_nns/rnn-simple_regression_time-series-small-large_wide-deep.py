'''
Generate a complete PyTorch NN code that satisfies the following four Requirements: 

Architecture: RNN-Simple 
Learning Task: Regression
Input Type + Scale: Time series, seq_length: Small <50 and features: Large >2000 
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
        self.rnn1 = nn.RNN(input_size=2048, hidden_size=128, num_layers=1, batch_first=True, nonlinearity='tanh', bias=True)
        self.rnn2 = nn.RNN(input_size=128, hidden_size=64, num_layers=1, batch_first=True, nonlinearity='tanh', bias=True)
        self.dropout = nn.Dropout(p=0.2)
        self.fc1 = nn.Linear(64, 32, bias=True)
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(32, 1, bias=True)

    def forward(self, x):
        out1, _ = self.rnn1(x)
        out1 = self.dropout(out1)
        out2, _ = self.rnn2(out1)
        out = out2[:, -1, :]
        out = self.fc1(out)
        out = self.act(out)
        out = self.fc2(out)
        return out