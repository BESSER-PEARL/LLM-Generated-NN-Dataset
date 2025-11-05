'''
Generate a complete PyTorch NN code that satisfies the following four Requirements: 

Architecture: RNN-Simple 
Learning Task: Classification-binary
Input Type + Scale: Time series, seq_length: Large >2000 and features: Small <50 
Complexity: Simple: Number of RNN-Simple layers up to 2, hidden_size of first RNN-Simple layer up to min(128, upper_bound_of_features).


- Provide complete PyTorch code for the model only.
- All layer parameters must be fixed values inside the network, with no variables or external constants used.
- Avoid creating excessively large layers that would cause unreasonable memory usage.
- The model must only use standard layers directly, without defining helper functions or custom block classes.
- Output only the code, with no comments.
'''
import torch
import torch.nn as nn

class RNNBinaryClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.rnn1 = nn.RNN(input_size=32, hidden_size=32, num_layers=1, nonlinearity='tanh', batch_first=True, bidirectional=False)
        self.rnn2 = nn.RNN(input_size=32, hidden_size=32, num_layers=1, nonlinearity='tanh', batch_first=True, bidirectional=False)
        self.fc = nn.Linear(32, 1)

    def forward(self, x):
        x, _ = self.rnn1(x)
        x, _ = self.rnn2(x)
        x = x[:, -1, :]
        x = self.fc(x)
        return x