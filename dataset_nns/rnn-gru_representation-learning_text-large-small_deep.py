'''
Generate a complete PyTorch NN code that satisfies the following four Requirements: 

Architecture: RNN-GRU 
Learning Task: Representation-learning
Input Type + Scale: Text, seq_length: Large >2000 and vocab_size: Small <1k 
Complexity: Deep: Number of RNN-GRU layers at least 2.


- Provide complete PyTorch code for the model only.
- All layer parameters must be fixed values inside the network, with no variables or external constants used.
- Avoid creating excessively large layers that would cause unreasonable memory usage.
- The model must only use standard layers directly, without defining helper functions or custom block classes.
- Output only the code, with no comments.
'''
import torch
import torch.nn as nn

class GRUTextEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(800, 64, padding_idx=0)
        self.dropout = nn.Dropout(0.2)
        self.gru = nn.GRU(input_size=64, hidden_size=128, num_layers=3, batch_first=True, dropout=0.2, bidirectional=False)
        self.proj1 = nn.Linear(384, 256)
        self.act = nn.ReLU()
        self.proj2 = nn.Linear(256, 128)
        self.norm = nn.LayerNorm(128)
    def forward(self, x):
        x = self.embedding(x)
        x = self.dropout(x)
        out, h = self.gru(x)
        last = h[-1]
        mean = out.mean(dim=1)
        max_vals, _ = out.max(dim=1)
        rep = torch.cat([last, mean, max_vals], dim=1)
        z = self.proj1(rep)
        z = self.act(z)
        z = self.dropout(z)
        z = self.proj2(z)
        z = self.norm(z)
        return z