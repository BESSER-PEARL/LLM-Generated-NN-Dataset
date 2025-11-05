'''
Generate a complete PyTorch NN code that satisfies the following four Requirements: 

Architecture: RNN-GRU 
Learning Task: Representation-learning
Input Type + Scale: Time series, seq_length: Large >2000 and features: Small <50 
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
        self.input_dropout = nn.Dropout(p=0.1)
        self.gru = nn.GRU(input_size=32, hidden_size=128, num_layers=3, batch_first=True, dropout=0.2, bidirectional=True)
        self.norm = nn.LayerNorm(256)
        self.proj1 = nn.Linear(512, 256)
        self.act = nn.GELU()
        self.drop = nn.Dropout(p=0.1)
        self.proj2 = nn.Linear(256, 128)
        self.out_norm = nn.LayerNorm(128)

    def forward(self, x):
        x = self.input_dropout(x)
        y, h = self.gru(x)
        y = self.norm(y)
        m = y.mean(dim=1)
        lf = h[-2]
        lb = h[-1]
        l = torch.cat((lf, lb), dim=1)
        r = torch.cat((m, l), dim=1)
        z = self.proj1(r)
        z = self.act(z)
        z = self.drop(z)
        z = self.proj2(z)
        z = self.out_norm(z)
        return z