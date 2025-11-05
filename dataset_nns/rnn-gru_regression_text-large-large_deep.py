'''
Generate a complete PyTorch NN code that satisfies the following four Requirements: 

Architecture: RNN-GRU 
Learning Task: Regression
Input Type + Scale: Text, seq_length: Large >2000 and vocab_size: Large >100k 
Complexity: Deep: Number of RNN-GRU layers at least 2.


- Provide complete PyTorch code for the model only.
- All layer parameters must be fixed values inside the network, with no variables or external constants used.
- Avoid creating excessively large layers that would cause unreasonable memory usage.
- The model must only use standard layers directly, without defining helper functions or custom block classes.
- Output only the code, with no comments.
'''
import torch
import torch.nn as nn

class DeepGRURegressor(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings=120000, embedding_dim=48, padding_idx=0)
        self.gru = nn.GRU(input_size=48, hidden_size=96, num_layers=3, batch_first=True, dropout=0.1, bidirectional=False)
        self.fc1 = nn.Linear(96, 64)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.1)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x):
        x = self.embedding(x)
        _, h_n = self.gru(x)
        h = h_n[-1]
        h = self.fc1(h)
        h = self.relu(h)
        h = self.dropout(h)
        y = self.fc2(h)
        return y