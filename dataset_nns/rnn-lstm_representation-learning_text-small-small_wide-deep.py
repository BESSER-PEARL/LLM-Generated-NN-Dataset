'''
Generate a complete PyTorch NN code that satisfies the following four Requirements: 

Architecture: RNN-LSTM 
Learning Task: Representation-learning
Input Type + Scale: Text, seq_length: Small <50 and vocab_size: Small <1k 
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
        self.embedding = nn.Embedding(800, 128, padding_idx=0)
        self.dropout = nn.Dropout(0.2)
        self.lstm = nn.LSTM(input_size=128, hidden_size=192, num_layers=2, batch_first=True, bidirectional=True, dropout=0.2)
        self.proj = nn.Linear(384, 128)
        self.norm = nn.LayerNorm(128)

    def forward(self, x):
        x = self.embedding(x)
        x = self.dropout(x)
        o, _ = self.lstm(x)
        m = o.mean(dim=1)
        r = self.proj(self.dropout(m))
        r = self.norm(r)
        return r