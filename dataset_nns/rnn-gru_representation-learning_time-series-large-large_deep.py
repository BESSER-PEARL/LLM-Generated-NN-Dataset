'''
Generate a complete PyTorch NN code that satisfies the following four Requirements: 

Architecture: RNN-GRU 
Learning Task: Representation-learning
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

class GRURepresentationNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_norm = nn.LayerNorm(2048)
        self.gru = nn.GRU(input_size=2048, hidden_size=256, num_layers=3, batch_first=True, dropout=0.2)
        self.post_norm = nn.LayerNorm(256)
        self.projection = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(256, 128)
        )

    def forward(self, x):
        x = self.input_norm(x)
        _, h = self.gru(x)
        h = h[-1]
        h = self.post_norm(h)
        z = self.projection(h)
        return z