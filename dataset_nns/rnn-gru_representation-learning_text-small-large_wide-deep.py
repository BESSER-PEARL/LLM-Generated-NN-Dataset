'''
Generate a complete PyTorch NN code that satisfies the following four Requirements: 

Architecture: RNN-GRU 
Learning Task: Representation-learning
Input Type + Scale: Text, seq_length: Small <50 and vocab_size: Large >100k 
Complexity: Wide-Deep: Number of RNN-GRU layers at least 2, embedding_dim of the Embedding layer at least min(128, upper_bound_of_vocab_size).


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
        self.embedding = nn.Embedding(120000, 128, padding_idx=0)
        self.dropout = nn.Dropout(0.1)
        self.gru = nn.GRU(128, 256, num_layers=3, batch_first=True, bidirectional=True, dropout=0.2)
        self.proj = nn.Linear(512, 256)
        self.norm = nn.LayerNorm(256)
        self.activation = nn.Tanh()
    def forward(self, x):
        x = self.embedding(x)
        x = self.dropout(x)
        _, h = self.gru(x)
        h = torch.cat((h[4], h[5]), dim=1)
        x = self.proj(h)
        x = self.norm(x)
        x = self.activation(x)
        return x