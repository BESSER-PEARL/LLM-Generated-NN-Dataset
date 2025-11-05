'''
Generate a complete PyTorch NN code that satisfies the following four Requirements: 

Architecture: RNN-GRU 
Learning Task: Representation-learning
Input Type + Scale: Text, seq_length: Small <50 and vocab_size: Small <1k 
Complexity: Deep: Number of RNN-GRU layers at least 2.


- Provide complete PyTorch code for the model only.
- All layer parameters must be fixed values inside the network, with no variables or external constants used.
- Avoid creating excessively large layers that would cause unreasonable memory usage.
- The model must only use standard layers directly, without defining helper functions or custom block classes.
- Output only the code, with no comments.
'''
import torch
import torch.nn as nn

class GRUTextRepresentation(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings=800, embedding_dim=64, padding_idx=0)
        self.gru = nn.GRU(input_size=64, hidden_size=128, num_layers=2, batch_first=True, dropout=0.2, bidirectional=False)
        self.layernorm = nn.LayerNorm(128)
        self.projection = nn.Linear(128, 64)
        self.output_norm = nn.LayerNorm(64)

    def forward(self, x):
        x = self.embedding(x)
        _, h_n = self.gru(x)
        rep = h_n[-1]
        rep = self.layernorm(rep)
        rep = torch.tanh(self.projection(rep))
        rep = self.output_norm(rep)
        return rep