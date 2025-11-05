'''
Generate a complete PyTorch NN code that satisfies the following four Requirements: 

Architecture: RNN-GRU 
Learning Task: Representation-learning
Input Type + Scale: Text, seq_length: Large >2000 and vocab_size: Small <1k 
Complexity: Wide-Deep: Number of RNN-GRU layers at least 2, embedding_dim of the Embedding layer at least min(128, upper_bound_of_vocab_size).


- Provide complete PyTorch code for the model only.
- All layer parameters must be fixed values inside the network, with no variables or external constants used.
- Avoid creating excessively large layers that would cause unreasonable memory usage.
- The model must only use standard layers directly, without defining helper functions or custom block classes.
- Output only the code, with no comments.
'''
import torch
import torch.nn as nn

class TextGRURepresentation(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(800, 128, padding_idx=0)
        self.gru = nn.GRU(input_size=128, hidden_size=192, num_layers=2, bidirectional=True, dropout=0.2, batch_first=True)
        self.proj = nn.Linear(384, 256)
        self.norm = nn.LayerNorm(256)
        self.dropout = nn.Dropout(0.2)
        self.act = nn.ReLU()
    def forward(self, x):
        x = self.embedding(x)
        _, h_n = self.gru(x)
        h = torch.cat((h_n[-2], h_n[-1]), dim=1)
        z = self.proj(h)
        z = self.norm(z)
        z = self.act(z)
        z = self.dropout(z)
        return z