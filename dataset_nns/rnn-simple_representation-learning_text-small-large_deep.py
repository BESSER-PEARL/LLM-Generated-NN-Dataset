'''
Generate a complete PyTorch NN code that satisfies the following four Requirements: 

Architecture: RNN-Simple 
Learning Task: Representation-learning
Input Type + Scale: Text, seq_length: Small <50 and vocab_size: Large >100k 
Complexity: Deep: Number of RNN-Simple layers at least 2.


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
        self.embedding = nn.Embedding(120000, 64, padding_idx=0)
        self.rnn = nn.RNN(input_size=64, hidden_size=128, num_layers=3, nonlinearity='tanh', batch_first=True, dropout=0.2, bidirectional=False)
        self.proj1 = nn.Linear(128, 192)
        self.act = nn.Tanh()
        self.drop = nn.Dropout(0.1)
        self.proj2 = nn.Linear(192, 128)
        self.norm = nn.LayerNorm(128, eps=1e-5, elementwise_affine=True)

    def forward(self, x):
        x = self.embedding(x)
        x, _ = self.rnn(x)
        x = x.mean(dim=1)
        x = self.proj1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.proj2(x)
        x = self.norm(x)
        return x