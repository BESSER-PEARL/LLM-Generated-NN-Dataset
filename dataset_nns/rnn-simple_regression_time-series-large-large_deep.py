'''
Generate a complete PyTorch NN code that satisfies the following four Requirements: 

Architecture: RNN-Simple 
Learning Task: Regression
Input Type + Scale: Time series, seq_length: Large >2000 and features: Large >2000 
Complexity: Deep: Number of RNN-Simple layers at least 2.


- Provide complete PyTorch code for the model only.
- All layer parameters must be fixed values inside the network, with no variables or external constants used.
- Avoid creating excessively large layers that would cause unreasonable memory usage.
- The model must only use standard layers directly, without defining helper functions or custom block classes.
- Output only the code, with no comments.
'''
import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.rnn1 = nn.RNN(input_size=2048, hidden_size=64, num_layers=1, nonlinearity='tanh', batch_first=True, bidirectional=False)
        self.rnn2 = nn.RNN(input_size=64, hidden_size=64, num_layers=1, nonlinearity='tanh', batch_first=True, bidirectional=False)
        self.fc = nn.Linear(64, 1)
    def forward(self, x):
        out1, _ = self.rnn1(x)
        out2, _ = self.rnn2(out1)
        y = self.fc(out2[:, -1, :])
        return y