'''
Generate a complete PyTorch NN code that satisfies the following four Requirements: 

Architecture: RNN-Simple 
Learning Task: Classification-multiclass
Input Type + Scale: Time series, seq_length: Small <50 and features: Small <50 
Complexity: Deep: Number of RNN-Simple layers at least 2.


- Provide complete PyTorch code for the model only.
- All layer parameters must be fixed values inside the network, with no variables or external constants used.
- Avoid creating excessively large layers that would cause unreasonable memory usage.
- The model must only use standard layers directly, without defining helper functions or custom block classes.
- Output only the code, with no comments.
'''
import torch
import torch.nn as nn

class TimeSeriesRNNClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.rnn = nn.RNN(input_size=24, hidden_size=64, num_layers=3, nonlinearity='tanh', batch_first=True, dropout=0.2, bidirectional=False)
        self.dropout1 = nn.Dropout(p=0.3)
        self.fc1 = nn.Linear(64, 64)
        self.act1 = nn.ReLU()
        self.dropout2 = nn.Dropout(p=0.3)
        self.fc2 = nn.Linear(64, 7)
    def forward(self, x):
        out, _ = self.rnn(x)
        y = out[:, -1, :]
        y = self.dropout1(y)
        y = self.fc1(y)
        y = self.act1(y)
        y = self.dropout2(y)
        y = self.fc2(y)
        return y