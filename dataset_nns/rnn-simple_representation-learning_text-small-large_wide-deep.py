'''
Generate a complete PyTorch NN code that satisfies the following four Requirements: 

Architecture: RNN-Simple 
Learning Task: Representation-learning
Input Type + Scale: Text, seq_length: Small <50 and vocab_size: Large >100k 
Complexity: Wide-Deep: Number of RNN-Simple layers at least 2, embedding_dim of the Embedding layer at least min(128, upper_bound_of_vocab_size).


- Provide complete PyTorch code for the model only.
- All layer parameters must be fixed values inside the network, with no variables or external constants used.
- Avoid creating excessively large layers that would cause unreasonable memory usage.
- The model must only use standard layers directly, without defining helper functions or custom block classes.
- Output only the code, with no comments.
'''
import torch
import torch.nn as nn

class RNNRepresentationModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(100357, 128)
        self.rnn1 = nn.RNN(128, 256, num_layers=1, nonlinearity='tanh', batch_first=True)
        self.dropout = nn.Dropout(0.2)
        self.rnn2 = nn.RNN(256, 256, num_layers=1, nonlinearity='tanh', batch_first=True)
        self.norm = nn.LayerNorm(256)
    def forward(self, x):
        x = self.embedding(x)
        x, _ = self.rnn1(x)
        x = self.dropout(x)
        x, _ = self.rnn2(x)
        x = x[:, -1, :]
        x = self.norm(x)
        return x