'''
Generate a complete PyTorch NN code that satisfies the following four Requirements: 

Architecture: RNN-GRU 
Learning Task: Representation-learning
Input Type + Scale: Text, seq_length: Large >2000 and vocab_size: Large >100k 
Complexity: Wide: embedding_dim of the Embedding layer at least min(128, upper_bound_of_vocab_size).


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
        self.embedding = nn.Embedding(num_embeddings=200000, embedding_dim=128, padding_idx=0)
        self.gru = nn.GRU(input_size=128, hidden_size=256, num_layers=1, batch_first=True, bidirectional=False)
        self.norm = nn.LayerNorm(256)
        self.proj = nn.Linear(256, 128)
        self.out_norm = nn.LayerNorm(128)

    def forward(self, x):
        x = self.embedding(x)
        _, h = self.gru(x)
        h = h[-1]
        h = self.norm(h)
        h = self.proj(h)
        h = self.out_norm(h)
        return h