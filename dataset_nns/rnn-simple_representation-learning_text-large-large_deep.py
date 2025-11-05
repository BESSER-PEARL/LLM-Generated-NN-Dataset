'''
Generate a complete PyTorch NN code that satisfies the following four Requirements: 

Architecture: RNN-Simple 
Learning Task: Representation-learning
Input Type + Scale: Text, seq_length: Large >2000 and vocab_size: Large >100k 
Complexity: Deep: Number of RNN-Simple layers at least 2.


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
        self.embedding = nn.Embedding(120000, 64, padding_idx=0)
        self.rnn1 = nn.RNN(input_size=64, hidden_size=128, num_layers=1, nonlinearity='tanh', batch_first=True, bias=True, dropout=0.0, bidirectional=False)
        self.dropout1 = nn.Dropout(p=0.1)
        self.rnn2 = nn.RNN(input_size=128, hidden_size=128, num_layers=1, nonlinearity='tanh', batch_first=True, bias=True, dropout=0.0, bidirectional=False)
        self.layernorm = nn.LayerNorm(128)
        self.proj = nn.Linear(128, 128, bias=True)
        self.out_norm = nn.LayerNorm(128)

    def forward(self, x):
        x = self.embedding(x)
        out1, _ = self.rnn1(x)
        out1 = self.dropout1(out1)
        out2, h2 = self.rnn2(out1)
        rep = h2[-1]
        rep = self.layernorm(rep)
        rep = self.proj(rep)
        rep = torch.tanh(rep)
        rep = self.out_norm(rep)
        return rep