'''
Generate a complete PyTorch NN code that satisfies the following four Requirements: 

Architecture: RNN-LSTM 
Learning Task: Representation-learning
Input Type + Scale: Text, seq_length: Small <50 and vocab_size: Small <1k 
Complexity: Deep: Number of RNN-LSTM layers at least 2.


- Provide complete PyTorch code for the model only.
- All layer parameters must be fixed values inside the network, with no variables or external constants used.
- Avoid creating excessively large layers that would cause unreasonable memory usage.
- The model must only use standard layers directly, without defining helper functions or custom block classes.
- Output only the code, with no comments.
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

class LSTMTextRepresentation(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(800, 64, padding_idx=0)
        self.lstm = nn.LSTM(input_size=64, hidden_size=128, num_layers=2, batch_first=True, dropout=0.1)
        self.projection = nn.Sequential(
            nn.Linear(128, 96),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(96, 64)
        )
        self.layer_norm = nn.LayerNorm(64)
    def forward(self, x):
        x = self.embedding(x)
        outputs, (h_n, c_n) = self.lstm(x)
        h = h_n[-1]
        z = self.projection(h)
        z = self.layer_norm(z)
        z = F.normalize(z, p=2, dim=1)
        return z