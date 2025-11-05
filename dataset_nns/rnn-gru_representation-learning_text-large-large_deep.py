'''
Generate a complete PyTorch NN code that satisfies the following four Requirements: 

Architecture: RNN-GRU 
Learning Task: Representation-learning
Input Type + Scale: Text, seq_length: Large >2000 and vocab_size: Large >100k 
Complexity: Deep: Number of RNN-GRU layers at least 2.


- Provide complete PyTorch code for the model only.
- All layer parameters must be fixed values inside the network, with no variables or external constants used.
- Avoid creating excessively large layers that would cause unreasonable memory usage.
- The model must only use standard layers directly, without defining helper functions or custom block classes.
- Output only the code, with no comments.
'''
import torch
import torch.nn as nn

class TextGRUEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings=120000, embedding_dim=32, padding_idx=0)
        self.gru = nn.GRU(input_size=32, hidden_size=64, num_layers=3, dropout=0.1, batch_first=True, bidirectional=False)
        self.proj = nn.Linear(64, 128)
        self.norm = nn.LayerNorm(128)
        self.act = nn.Tanh()
    def forward(self, x):
        e = self.embedding(x)
        _, h_n = self.gru(e)
        h = h_n[-1]
        z = self.proj(h)
        z = self.norm(z)
        z = self.act(z)
        return z