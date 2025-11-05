'''
Generate a complete PyTorch NN code that satisfies the following four Requirements: 

Architecture: RNN-Simple 
Learning Task: Regression
Input Type + Scale: Text, seq_length: Small <50 and vocab_size: Large >100k 
Complexity: Simple: Number of RNN-Simple layers up to 2, embedding_dim of the Embedding layer up to min(128, upper_bound_of_vocab_size).


- Provide complete PyTorch code for the model only.
- All layer parameters must be fixed values inside the network, with no variables or external constants used.
- Avoid creating excessively large layers that would cause unreasonable memory usage.
- The model must only use standard layers directly, without defining helper functions or custom block classes.
- Output only the code, with no comments.
'''
import torch
import torch.nn as nn

class SimpleTextRNNRegressor(nn.Module):
    def __init__(self):
        super(SimpleTextRNNRegressor, self).__init__()
        self.embedding = nn.Embedding(num_embeddings=120000, embedding_dim=128)
        self.rnn = nn.RNN(input_size=128, hidden_size=64, num_layers=2, nonlinearity='tanh', batch_first=True, dropout=0.0, bidirectional=False)
        self.output = nn.Linear(64, 1)

    def forward(self, x):
        x = self.embedding(x)
        x, _ = self.rnn(x)
        x = x[:, -1, :]
        x = self.output(x)
        return x