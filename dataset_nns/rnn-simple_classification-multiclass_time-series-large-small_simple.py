'''
Generate a complete PyTorch NN code that satisfies the following four Requirements: 

Architecture: RNN-Simple 
Learning Task: Classification-multiclass
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

class RNNClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.rnn = nn.RNN(input_size=32, hidden_size=32, num_layers=2, nonlinearity='tanh', batch_first=True, dropout=0.0, bidirectional=False)
        self.fc = nn.Linear(32, 10)
    def forward(self, x):
        _, h_n = self.rnn(x)
        x = h_n[-1]
        x = self.fc(x)
        return x