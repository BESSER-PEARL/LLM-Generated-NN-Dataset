'''
Generate a complete PyTorch NN code that satisfies the following four Requirements: 

Architecture: RNN-Simple 
Learning Task: Representation-learning
Input Type + Scale: Text, seq_length: Large >2000 and vocab_size: Small <1k 
Complexity: Deep: Number of RNN-Simple layers at least 2.


- Provide complete PyTorch code for the model only.
- All layer parameters must be fixed values inside the network, with no variables or external constants used.
- Avoid creating excessively large layers that would cause unreasonable memory usage.
- The model must only use standard layers directly, without defining helper functions or custom block classes.
- Output only the code, with no comments.
'''
import torch
import torch.nn as nn

class RnnSimpleRepresentationModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(512, 128, padding_idx=0)
        self.rnn = nn.RNN(128, 96, num_layers=2, nonlinearity='tanh', batch_first=True, dropout=0.1)
        self.projection = nn.Linear(96, 64)
        self.norm = nn.LayerNorm(64)

    def forward(self, x):
        x = self.embedding(x)
        _, h = self.rnn(x)
        h = h[-1]
        x = self.projection(h)
        x = self.norm(x)
        return x