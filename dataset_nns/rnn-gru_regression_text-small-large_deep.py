'''
Generate a complete PyTorch NN code that satisfies the following four Requirements: 

Architecture: RNN-GRU 
Learning Task: Regression
Input Type + Scale: Text, seq_length: Small <50 and vocab_size: Large >100k 
Complexity: Deep: Number of RNN-GRU layers at least 2.


- Provide complete PyTorch code for the model only.
- All layer parameters must be fixed values inside the network, with no variables or external constants used.
- Avoid creating excessively large layers that would cause unreasonable memory usage.
- The model must only use standard layers directly, without defining helper functions or custom block classes.
- Output only the code, with no comments.
'''
import torch
import torch.nn as nn

class TextGRURegressor(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings=120000, embedding_dim=64, padding_idx=0)
        self.gru = nn.GRU(input_size=64, hidden_size=128, num_layers=3, batch_first=True, dropout=0.2, bidirectional=False)
        self.dropout = nn.Dropout(p=0.1)
        self.fc = nn.Linear(128, 1)

    def forward(self, x):
        x = self.embedding(x)
        _, h = self.gru(x)
        h_last = h[-1]
        h_last = self.dropout(h_last)
        y = self.fc(h_last)
        return y