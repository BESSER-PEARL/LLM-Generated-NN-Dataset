'''
Generate a complete PyTorch NN code that satisfies the following four Requirements: 

Architecture: RNN-LSTM 
Learning Task: Representation-learning
Input Type + Scale: Text, seq_length: Large >2000 and vocab_size: Large >100k 
Complexity: Deep: Number of RNN-LSTM layers at least 2.


- Provide complete PyTorch code for the model only.
- All layer parameters must be fixed values inside the network, with no variables or external constants used.
- Avoid creating excessively large layers that would cause unreasonable memory usage.
- The model must only use standard layers directly, without defining helper functions or custom block classes.
- Output only the code, with no comments.
'''
import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(120000, 64, padding_idx=0)
        self.lstm = nn.LSTM(64, 256, num_layers=3, dropout=0.2, batch_first=True)
        self.norm = nn.LayerNorm(256)
        self.dropout = nn.Dropout(0.2)
        self.proj = nn.Linear(256, 128)

    def forward(self, x):
        x = self.embedding(x)
        x, (h, c) = self.lstm(x)
        x = h[-1]
        x = self.norm(x)
        x = self.dropout(x)
        x = self.proj(x)
        return x