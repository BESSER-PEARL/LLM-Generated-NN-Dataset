'''
Generate a complete PyTorch NN code that satisfies the following four Requirements: 

Architecture: RNN-GRU 
Learning Task: Classification-binary
Input Type + Scale: Time series, seq_length: Small <50 and features: Large >2000 
Complexity: Deep: Number of RNN-GRU layers at least 2.


- Provide complete PyTorch code for the model only.
- All layer parameters must be fixed values inside the network, with no variables or external constants used.
- Avoid creating excessively large layers that would cause unreasonable memory usage.
- The model must only use standard layers directly, without defining helper functions or custom block classes.
- Output only the code, with no comments.
'''
import torch
import torch.nn as nn

class GRUBinaryClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.gru = nn.GRU(input_size=2048, hidden_size=256, num_layers=3, batch_first=True, dropout=0.2)
        self.dropout = nn.Dropout(0.2)
        self.fc1 = nn.Linear(256, 128)
        self.relu = nn.ReLU()
        self.fc_out = nn.Linear(128, 1)

    def forward(self, x):
        out, _ = self.gru(x)
        last = out[:, -1, :]
        last = self.dropout(last)
        z = self.fc1(last)
        z = self.relu(z)
        z = self.dropout(z)
        logit = self.fc_out(z)
        return logit