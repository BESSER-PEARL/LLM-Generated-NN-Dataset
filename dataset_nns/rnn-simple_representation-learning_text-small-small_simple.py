'''
Generate a complete PyTorch NN code that satisfies the following four Requirements: 

Architecture: RNN-Simple 
Learning Task: Representation-learning
Input Type + Scale: Text, seq_length: Small <50 and vocab_size: Small <1k 
Complexity: Simple: Number of RNN-Simple layers up to 2, embedding_dim of the Embedding layer up to min(128, upper_bound_of_vocab_size).


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
        self.embedding = nn.Embedding(800, 64)
        self.rnn = nn.RNN(64, 96, num_layers=2, batch_first=True, dropout=0.1, nonlinearity='tanh')
        self.proj = nn.Linear(96, 64)
        self.norm = nn.LayerNorm(64)

    def forward(self, x):
        x = self.embedding(x)
        _, h_n = self.rnn(x)
        h_top = h_n[-1]
        z = self.proj(h_top)
        z = self.norm(z)
        return z