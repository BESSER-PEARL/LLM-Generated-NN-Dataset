'''
Generate a complete PyTorch NN code that satisfies the following four Requirements: 

Architecture: RNN-LSTM 
Learning Task: Regression
Input Type + Scale: Time series, seq_length: Large >2000 and features: Small <50 
Complexity: Wide: hidden_size of first RNN-LSTM layer at least min(128, upper_bound_of_features).


- Provide complete PyTorch code for the model only.
- All layer parameters must be fixed values inside the network, with no variables or external constants used.
- Avoid creating excessively large layers that would cause unreasonable memory usage.
- The model must only use standard layers directly, without defining helper functions or custom block classes.
- Output only the code, with no comments.
'''
import torch
import torch.nn as nn

class RNNLSTMRegressor(nn.Module):
    def __init__(self):
        super().__init__()
        self.rnn = nn.RNN(input_size=32, hidden_size=64, num_layers=1, nonlinearity='tanh', batch_first=True, bias=True, dropout=0.0, bidirectional=False)
        self.lstm = nn.LSTM(input_size=64, hidden_size=64, num_layers=1, batch_first=True, bias=True, dropout=0.0, bidirectional=False)
        self.fc1 = nn.Linear(64, 16, bias=True)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.1)
        self.out = nn.Linear(16, 1, bias=True)

    def forward(self, x):
        x, _ = self.rnn(x)
        x, _ = self.lstm(x)
        x = x[:, -1, :]
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.out(x)
        return x