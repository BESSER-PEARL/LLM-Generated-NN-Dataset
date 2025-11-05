'''
Generate a complete PyTorch NN code that satisfies the following four Requirements: 

Architecture: RNN-Simple 
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

class SimpleRNNTextRepresentation(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings=131072, embedding_dim=128, padding_idx=0)
        self.rnn = nn.RNN(input_size=128, hidden_size=256, num_layers=1, nonlinearity='tanh', batch_first=True, bias=True, dropout=0.0, bidirectional=False)
        self.proj = nn.Linear(256, 128, bias=True)
        self.norm = nn.LayerNorm(128)
    def forward(self, x):
        x = self.embedding(x)
        _, h_n = self.rnn(x)
        x = self.proj(h_n[-1])
        x = self.norm(x)
        return x