'''
Generate a complete PyTorch NN code that satisfies the following four Requirements: 

Architecture: RNN-LSTM 
Learning Task: Representation-learning
Input Type + Scale: Text, seq_length: Small <50 and vocab_size: Small <1k 
Complexity: Simple: Number of RNN-LSTM layers up to 2, embedding_dim of the Embedding layer up to min(128, upper_bound_of_vocab_size).


- Provide complete PyTorch code for the model only.
- All layer parameters must be fixed values inside the network, with no variables or external constants used.
- Avoid creating excessively large layers that would cause unreasonable memory usage.
- The model must only use standard layers directly, without defining helper functions or custom block classes.
- Output only the code, with no comments.
'''
import torch
import torch.nn as nn

class TextLSTMEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings=800, embedding_dim=64, padding_idx=0)
        self.lstm = nn.LSTM(input_size=64, hidden_size=64, num_layers=2, batch_first=True, dropout=0.1)
        self.proj = nn.Linear(64, 32)
        self.act = nn.Tanh()
        self.norm = nn.LayerNorm(32)

    def forward(self, x):
        x = self.embedding(x)
        output, (h_n, c_n) = self.lstm(x)
        rep = h_n[-1]
        rep = self.proj(rep)
        rep = self.act(rep)
        rep = self.norm(rep)
        return rep