'''
Generate a complete PyTorch NN code that satisfies the following four Requirements: 

Architecture: RNN-GRU 
Learning Task: Classification-binary
Input Type + Scale: Time series, seq_length: Large >2000 and features: Small <50 
Complexity: Wide: hidden_size of first RNN-GRU layer at least min(128, upper_bound_of_features).


- Provide complete PyTorch code for the model only.
- All layer parameters must be fixed values inside the network, with no variables or external constants used.
- Avoid creating excessively large layers that would cause unreasonable memory usage.
- The model must only use standard layers directly, without defining helper functions or custom block classes.
- Output only the code, with no comments.
'''
import torch
import torch.nn as nn

class GRUTimeSeriesBinaryClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.gru = nn.GRU(input_size=32, hidden_size=64, num_layers=2, batch_first=True, dropout=0.2)
        self.dropout = nn.Dropout(p=0.3)
        self.fc1 = nn.Linear(64, 32)
        self.relu = nn.ReLU()
        self.out = nn.Linear(32, 1)

    def forward(self, x):
        _, h = self.gru(x)
        x = h[-1]
        x = self.dropout(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.out(x)
        return x