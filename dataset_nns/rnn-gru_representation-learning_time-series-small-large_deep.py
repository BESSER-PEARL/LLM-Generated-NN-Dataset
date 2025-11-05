'''
Generate a complete PyTorch NN code that satisfies the following four Requirements: 

Architecture: RNN-GRU 
Learning Task: Representation-learning
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

class GRURepresentationModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.gru = nn.GRU(input_size=2048, hidden_size=192, num_layers=3, batch_first=True, dropout=0.2, bidirectional=False)
        self.norm = nn.LayerNorm(192)
        self.dropout = nn.Dropout(0.1)
        self.proj = nn.Linear(192, 128)
        self.act = nn.Tanh()
    def forward(self, x):
        _, h = self.gru(x)
        x = h[-1]
        x = self.norm(x)
        x = self.dropout(x)
        x = self.proj(x)
        x = self.act(x)
        return x