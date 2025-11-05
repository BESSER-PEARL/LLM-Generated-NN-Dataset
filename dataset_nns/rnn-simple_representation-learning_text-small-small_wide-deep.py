'''
Generate a complete PyTorch NN code that satisfies the following four Requirements: 

Architecture: RNN-Simple 
Learning Task: Representation-learning
Input Type + Scale: Text, seq_length: Small <50 and vocab_size: Small <1k 
Complexity: Wide-Deep: Number of RNN-Simple layers at least 2, embedding_dim of the Embedding layer at least min(128, upper_bound_of_vocab_size).


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
        super(TextRNNRepresentation, self).__init__()
        self.embedding = nn.Embedding(800, 256, padding_idx=0)
        self.rnn = nn.RNN(input_size=256, hidden_size=256, num_layers=2, nonlinearity='tanh', batch_first=True, dropout=0.2, bidirectional=False)
        self.dropout = nn.Dropout(0.2)
        self.proj = nn.Linear(512, 128)
        self.norm = nn.LayerNorm(128)

    def forward(self, x):
        x = self.embedding(x)
        out, h = self.rnn(x)
        last = h[-1]
        mean_pooled = out.mean(dim=1)
        features = torch.cat([last, mean_pooled], dim=1)
        features = self.dropout(features)
        z = self.proj(features)
        z = self.norm(z)
        z = torch.tanh(z)
        return z