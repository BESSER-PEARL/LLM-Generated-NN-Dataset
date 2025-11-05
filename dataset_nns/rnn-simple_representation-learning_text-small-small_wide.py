'''
Generate a complete PyTorch NN code that satisfies the following four Requirements: 

Architecture: RNN-Simple 
Learning Task: Representation-learning
Input Type + Scale: Text, seq_length: Small <50 and vocab_size: Small <1k 
Complexity: Wide: embedding_dim of the Embedding layer at least min(128, upper_bound_of_vocab_size).


- Provide complete PyTorch code for the model only.
- All layer parameters must be fixed values inside the network, with no variables or external constants used.
- Avoid creating excessively large layers that would cause unreasonable memory usage.
- The model must only use standard layers directly, without defining helper functions or custom block classes.
- Output only the code, with no comments.
'''
import torch
import torch.nn as nn

class TextRNNRepresentation(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings=800, embedding_dim=128, padding_idx=0)
        self.dropout = nn.Dropout(p=0.2)
        self.rnn = nn.RNN(input_size=128, hidden_size=128, num_layers=1, nonlinearity='tanh', batch_first=True, bidirectional=False)
        self.proj = nn.Sequential(
            nn.Linear(128, 128, bias=True),
            nn.LayerNorm(128),
            nn.Tanh()
        )
    def forward(self, x):
        x = self.embedding(x)
        x = self.dropout(x)
        _, h = self.rnn(x)
        h = h[-1]
        z = self.proj(h)
        return z