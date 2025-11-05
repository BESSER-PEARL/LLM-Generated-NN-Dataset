'''
Generate a complete PyTorch NN code that satisfies the following four Requirements: 

Architecture: RNN-GRU 
Learning Task: Representation-learning
Input Type + Scale: Text, seq_length: Small <50 and vocab_size: Large >100k 
Complexity: Wide: embedding_dim of the Embedding layer at least min(128, upper_bound_of_vocab_size).


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
        self.embedding = nn.Embedding(num_embeddings=150000, embedding_dim=128, padding_idx=0)
        self.dropout = nn.Dropout(p=0.1)
        self.gru = nn.GRU(input_size=128, hidden_size=256, num_layers=1, batch_first=True, bidirectional=True)
        self.proj = nn.Linear(in_features=512, out_features=256)
        self.norm = nn.LayerNorm(normalized_shape=256)
        self.act = nn.Tanh()
    def forward(self, x):
        x = self.embedding(x)
        x = self.dropout(x)
        _, h = self.gru(x)
        x = torch.cat((h[0], h[1]), dim=1)
        x = self.proj(x)
        x = self.act(x)
        x = self.norm(x)
        return x