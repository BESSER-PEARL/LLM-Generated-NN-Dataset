'''
Generate a complete PyTorch NN code that satisfies the following four Requirements: 

Architecture: RNN-GRU 
Learning Task: Representation-learning
Input Type + Scale: Time series, seq_length: Small <50 and features: Small <50 
Complexity: Simple: Number of RNN-GRU layers up to 2, hidden_size of first RNN-GRU layer up to min(128, upper_bound_of_features).


- Provide complete PyTorch code for the model only.
- All layer parameters must be fixed values inside the network, with no variables or external constants used.
- Avoid creating excessively large layers that would cause unreasonable memory usage.
- The model must only use standard layers directly, without defining helper functions or custom block classes.
- Output only the code, with no comments.
'''
import torch
import torch.nn as nn

class TimeSeriesGRUEncoder(nn.Module):
    def __init__(self):
        super(TimeSeriesGRUEncoder, self).__init__()
        self.gru1 = nn.GRU(input_size=24, hidden_size=24, num_layers=1, batch_first=True)
        self.gru2 = nn.GRU(input_size=24, hidden_size=16, num_layers=1, batch_first=True)
        self.dropout = nn.Dropout(p=0.2)
        self.fc = nn.Linear(in_features=16, out_features=12)
        self.activation = nn.Tanh()
    def forward(self, x):
        x, _ = self.gru1(x)
        x, h2 = self.gru2(x)
        h = h2[-1]
        h = self.dropout(h)
        z = self.fc(h)
        z = self.activation(z)
        return z