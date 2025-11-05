'''
Generate a complete PyTorch NN code that satisfies the following four Requirements: 

Architecture: RNN-LSTM 
Learning Task: Representation-learning
Input Type + Scale: Text, seq_length: Large >2000 and vocab_size: Small <1k 
Complexity: Wide-Deep: Number of RNN-LSTM layers at least 2, embedding_dim of the Embedding layer at least min(128, upper_bound_of_vocab_size).


- Provide complete PyTorch code for the model only.
- All layer parameters must be fixed values inside the network, with no variables or external constants used.
- Avoid creating excessively large layers that would cause unreasonable memory usage.
- The model must only use standard layers directly, without defining helper functions or custom block classes.
- Output only the code, with no comments.
'''
import torch
import torch.nn as nn

class TextRepLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(800, 192, padding_idx=0)
        self.enc_dropout = nn.Dropout(0.1)
        self.lstm1 = nn.LSTM(192, 160, num_layers=1, batch_first=True, bidirectional=True)
        self.lstm2 = nn.LSTM(320, 160, num_layers=1, batch_first=True, bidirectional=True)
        self.norm = nn.LayerNorm(320)
        self.pool_dropout = nn.Dropout(0.1)
        self.proj = nn.Linear(640, 128)
        self.out_norm = nn.LayerNorm(128)

    def forward(self, x):
        x = self.embedding(x)
        x = self.enc_dropout(x)
        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x)
        x = self.norm(x)
        mean_pool = x.mean(dim=1)
        max_pool, _ = x.max(dim=1)
        x = torch.cat([mean_pool, max_pool], dim=1)
        x = self.pool_dropout(x)
        x = self.proj(x)
        x = torch.tanh(x)
        x = self.out_norm(x)
        return x