'''
Generate a complete PyTorch NN code that satisfies the following four Requirements: 

Architecture: RNN-Simple 
Learning Task: Representation-learning
Input Type + Scale: Text, seq_length: Small <50 and vocab_size: Small <1k 
Complexity: Deep: Number of RNN-Simple layers at least 2.


- Provide complete PyTorch code for the model only.
- All layer parameters must be fixed values inside the network, with no variables or external constants used.
- Avoid creating excessively large layers that would cause unreasonable memory usage.
- The model must only use standard layers directly, without defining helper functions or custom block classes.
- Output only the code, with no comments.
'''
import torch
import torch.nn as nn

class TextRNNEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings=800, embedding_dim=64, padding_idx=0)
        self.rnn = nn.RNN(input_size=64, hidden_size=128, num_layers=3, nonlinearity='tanh', batch_first=True, dropout=0.1, bidirectional=False)
        self.dropout = nn.Dropout(p=0.2)
        self.projection = nn.Linear(128, 64)
        self.norm = nn.LayerNorm(64)

    def forward(self, x):
        x = self.embedding(x)
        _, h_n = self.rnn(x)
        x = h_n[-1]
        x = self.dropout(x)
        x = self.projection(x)
        x = self.norm(x)
        return x