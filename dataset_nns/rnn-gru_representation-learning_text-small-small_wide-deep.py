'''
Generate a complete PyTorch NN code that satisfies the following four Requirements: 

Architecture: RNN-GRU 
Learning Task: Representation-learning
Input Type + Scale: Text, seq_length: Small <50 and vocab_size: Small <1k 
Complexity: Wide-Deep: Number of RNN-GRU layers at least 2, embedding_dim of the Embedding layer at least min(128, upper_bound_of_vocab_size).


- Provide complete PyTorch code for the model only.
- All layer parameters must be fixed values inside the network, with no variables or external constants used.
- Avoid creating excessively large layers that would cause unreasonable memory usage.
- The model must only use standard layers directly, without defining helper functions or custom block classes.
- Output only the code, with no comments.
'''
import torch
import torch.nn as nn

class RNNGRURepresentationModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(800, 128, padding_idx=0)
        self.dropout = nn.Dropout(0.1)
        self.gru = nn.GRU(input_size=128, hidden_size=192, num_layers=3, batch_first=True, dropout=0.2, bidirectional=True)
        self.norm = nn.LayerNorm(384)
        self.fc1 = nn.Linear(384, 256)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(256, 128)
    def forward(self, x):
        x = self.embedding(x)
        x = self.dropout(x)
        _, h_n = self.gru(x)
        h = torch.cat((h_n[-2], h_n[-1]), dim=1)
        h = self.norm(h)
        h = self.fc1(h)
        h = self.relu(h)
        h = self.fc2(h)
        return h