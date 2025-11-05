'''
Generate a complete PyTorch NN code that satisfies the following four Requirements: 

Architecture: RNN-Simple 
Learning Task: Regression
Input Type + Scale: Text, seq_length: Small <50 and vocab_size: Large >100k 
Complexity: Wide: embedding_dim of the Embedding layer at least min(128, upper_bound_of_vocab_size).


- Provide complete PyTorch code for the model only.
- All layer parameters must be fixed values inside the network, with no variables or external constants used.
- Avoid creating excessively large layers that would cause unreasonable memory usage.
- The model must only use standard layers directly, without defining helper functions or custom block classes.
- Output only the code, with no comments.
'''
import torch
import torch.nn as nn

class RNNTextRegressor(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings=120000, embedding_dim=128)
        self.rnn = nn.RNN(input_size=128, hidden_size=64, num_layers=1, nonlinearity='tanh', batch_first=True)
        self.fc = nn.Linear(in_features=64, out_features=1)

    def forward(self, x):
        x = self.embedding(x)
        _, h_n = self.rnn(x)
        y = self.fc(h_n.squeeze(0))
        return y