'''
Generate a complete PyTorch NN code that satisfies the following four Requirements: 

Architecture: RNN-GRU 
Learning Task: Classification-binary
Input Type + Scale: Time series, seq_length: Large >2000 and features: Large >2000 
Complexity: Deep: Number of RNN-GRU layers at least 2.


- Provide complete PyTorch code for the model only.
- All layer parameters must be fixed values inside the network, with no variables or external constants used.
- Avoid creating excessively large layers that would cause unreasonable memory usage.
- The model must only use standard layers directly, without defining helper functions or custom block classes.
- Output only the code, with no comments.
'''
import torch
import torch.nn as nn

class GRUClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.gru = nn.GRU(input_size=2048, hidden_size=64, num_layers=2, batch_first=True, dropout=0.1, bidirectional=False)
        self.fc = nn.Linear(64, 1)
    def forward(self, x):
        _, h_n = self.gru(x)
        x = h_n[-1]
        x = self.fc(x)
        return x