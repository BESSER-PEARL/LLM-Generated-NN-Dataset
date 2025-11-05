'''
Generate a complete PyTorch NN code that satisfies the following four Requirements: 

Architecture: RNN-LSTM 
Learning Task: Representation-learning
Input Type + Scale: Text, seq_length: Large >2000 and vocab_size: Small <1k 
Complexity: Deep: Number of RNN-LSTM layers at least 2.


- Provide complete PyTorch code for the model only.
- All layer parameters must be fixed values inside the network, with no variables or external constants used.
- Avoid creating excessively large layers that would cause unreasonable memory usage.
- The model must only use standard layers directly, without defining helper functions or custom block classes.
- Output only the code, with no comments.
'''
import torch
import torch.nn as nn

class DeepLSTMTextEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(800, 128, padding_idx=0)
        self.lstm = nn.LSTM(input_size=128, hidden_size=192, num_layers=2, batch_first=True, dropout=0.2, bidirectional=True)
        self.norm = nn.LayerNorm(384)
        self.dropout = nn.Dropout(0.1)
        self.proj = nn.Linear(384, 128)
        self.out_norm = nn.LayerNorm(128)

    def forward(self, x):
        x = self.embedding(x)
        x, _ = self.lstm(x)
        x = self.norm(x)
        x = self.dropout(x)
        x = x.mean(dim=1)
        x = self.proj(x)
        x = self.out_norm(x)
        return x