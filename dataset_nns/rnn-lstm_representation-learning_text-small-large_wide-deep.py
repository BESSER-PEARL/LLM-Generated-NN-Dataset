'''
Generate a complete PyTorch NN code that satisfies the following four Requirements: 

Architecture: RNN-LSTM 
Learning Task: Representation-learning
Input Type + Scale: Text, seq_length: Small <50 and vocab_size: Large >100k 
Complexity: Wide-Deep: Number of RNN-LSTM layers at least 2, embedding_dim of the Embedding layer at least min(128, upper_bound_of_vocab_size).


- Provide complete PyTorch code for the model only.
- All layer parameters must be fixed values inside the network, with no variables or external constants used.
- Avoid creating excessively large layers that would cause unreasonable memory usage.
- The model must only use standard layers directly, without defining helper functions or custom block classes.
- Output only the code, with no comments.
'''
import torch
import torch.nn as nn

class WideDeepLSTMRepresentation(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(120000, 128, padding_idx=0)
        self.emb_dropout = nn.Dropout(0.1)
        self.lstm = nn.LSTM(input_size=128, hidden_size=256, num_layers=2, batch_first=True, bidirectional=True, dropout=0.2)
        self.norm = nn.LayerNorm(512)
        self.proj = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.LayerNorm(256)
        )

    def forward(self, x):
        x = self.embedding(x)
        x = self.emb_dropout(x)
        _, (h, _) = self.lstm(x)
        h_fwd = h[2]
        h_bwd = h[3]
        rep = torch.cat([h_fwd, h_bwd], dim=1)
        rep = self.norm(rep)
        rep = self.proj(rep)
        return rep