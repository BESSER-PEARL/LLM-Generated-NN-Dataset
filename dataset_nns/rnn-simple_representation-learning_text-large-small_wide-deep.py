'''
Generate a complete PyTorch NN code that satisfies the following four Requirements: 

Architecture: RNN-Simple 
Learning Task: Representation-learning
Input Type + Scale: Text, seq_length: Large >2000 and vocab_size: Small <1k 
Complexity: Wide-Deep: Number of RNN-Simple layers at least 2, embedding_dim of the Embedding layer at least min(128, upper_bound_of_vocab_size).


- Provide complete PyTorch code for the model only.
- All layer parameters must be fixed values inside the network, with no variables or external constants used.
- Avoid creating excessively large layers that would cause unreasonable memory usage.
- The model must only use standard layers directly, without defining helper functions or custom block classes.
- Output only the code, with no comments.
'''
import torch
from torch import nn

class TextRNNRepresentation(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings=800, embedding_dim=128, padding_idx=0)
        self.rnn = nn.RNN(input_size=128, hidden_size=192, num_layers=2, batch_first=True, dropout=0.1, bidirectional=True, nonlinearity='tanh')
        self.dropout = nn.Dropout(p=0.1)
        self.proj = nn.Linear(768, 256)
        self.norm = nn.LayerNorm(256)
    def forward(self, x):
        x = self.embedding(x)
        x = self.dropout(x)
        out, _ = self.rnn(x)
        pooled = out.mean(dim=1)
        last = out[:, -1, :]
        rep = torch.cat([pooled, last], dim=1)
        rep = self.proj(rep)
        rep = torch.tanh(self.norm(rep))
        return rep