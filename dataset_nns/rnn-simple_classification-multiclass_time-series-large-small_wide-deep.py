'''
Generate a complete PyTorch NN code that satisfies the following four Requirements: 

Architecture: RNN-Simple 
Learning Task: Classification-multiclass
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

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.rnn_deep = nn.RNN(input_size=32, hidden_size=64, num_layers=2, batch_first=True, nonlinearity='tanh', dropout=0.0, bidirectional=False)
        self.rnn_wide = nn.RNN(input_size=32, hidden_size=32, num_layers=1, batch_first=True, nonlinearity='relu', dropout=0.0, bidirectional=False)
        self.dropout = nn.Dropout(p=0.2)
        self.fc1 = nn.Linear(96, 64)
        self.relu = nn.ReLU()
        self.fc_out = nn.Linear(64, 10)

    def forward(self, x):
        out_deep, _ = self.rnn_deep(x)
        out_wide, _ = self.rnn_wide(x)
        last_deep = out_deep[:, -1, :]
        last_wide = out_wide[:, -1, :]
        combined = torch.cat([last_deep, last_wide], dim=1)
        z = self.dropout(self.relu(self.fc1(combined)))
        logits = self.fc_out(z)
        return logits