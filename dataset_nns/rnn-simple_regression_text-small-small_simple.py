'''
Generate a complete PyTorch NN code that satisfies the following four Requirements: 

Architecture: RNN-Simple 
Learning Task: Regression
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

class TextRNNRegression(nn.Module):
    def __init__(self):
        super(TextRNNRegression, self).__init__()
        self.embedding = nn.Embedding(500, 64, padding_idx=0)
        self.rnn = nn.RNN(64, 64, num_layers=2, batch_first=True, nonlinearity='tanh', bias=True)
        self.fc = nn.Linear(64, 1)
    def forward(self, x):
        x = self.embedding(x)
        _, h = self.rnn(x)
        x = h[-1]
        x = self.fc(x)
        return x