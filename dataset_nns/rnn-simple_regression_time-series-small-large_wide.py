'''
Generate a complete PyTorch NN code that satisfies the following four Requirements: 

Architecture: RNN-Simple 
Learning Task: Regression
Input Type + Scale: Time series, seq_length: Small <50 and features: Large >2000 
Complexity: Wide: hidden_size of first RNN-Simple layer at least min(128, upper_bound_of_features).


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
        self.rnn = nn.RNN(input_size=2048, hidden_size=256, num_layers=1, batch_first=True, nonlinearity='tanh', bidirectional=False)
        self.fc1 = nn.Linear(256, 64)
        self.act = nn.ReLU()
        self.fc_out = nn.Linear(64, 1)

    def forward(self, x):
        out, _ = self.rnn(x)
        last = out[:, -1, :]
        y = self.fc1(last)
        y = self.act(y)
        y = self.fc_out(y)
        return y