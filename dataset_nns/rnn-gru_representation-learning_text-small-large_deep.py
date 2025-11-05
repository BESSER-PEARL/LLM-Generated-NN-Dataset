'''
Generate a complete PyTorch NN code that satisfies the following four Requirements: 

Architecture: RNN-GRU 
Learning Task: Representation-learning
Input Type + Scale: Text, seq_length: Small <50 and vocab_size: Large >100k 
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
        self.embedding = nn.Embedding(num_embeddings=120000, embedding_dim=128, padding_idx=0)
        self.gru = nn.GRU(input_size=128, hidden_size=256, num_layers=3, batch_first=True, dropout=0.25)
        self.projection = nn.Linear(in_features=256, out_features=128)
        self.norm = nn.LayerNorm(normalized_shape=128, eps=1e-5)

    def forward(self, x):
        x = self.embedding(x)
        _, h_n = self.gru(x)
        x = h_n[-1]
        x = self.projection(x)
        x = self.norm(x)
        return x