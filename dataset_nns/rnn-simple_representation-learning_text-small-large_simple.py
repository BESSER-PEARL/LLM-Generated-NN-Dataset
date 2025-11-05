'''
Generate a complete PyTorch NN code that satisfies the following four Requirements: 

Architecture: RNN-Simple 
Learning Task: Representation-learning
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

class TextRNNRepresentation(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(100003, 96, padding_idx=0)
        self.rnn = nn.RNN(input_size=96, hidden_size=128, num_layers=2, nonlinearity='tanh', batch_first=True, dropout=0.0, bidirectional=False)
    def forward(self, x):
        emb = self.embedding(x)
        _, h_n = self.rnn(emb)
        return h_n[-1]