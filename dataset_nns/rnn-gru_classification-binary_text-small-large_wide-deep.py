'''
Generate a complete PyTorch NN code that satisfies the following four Requirements: 

Architecture: RNN-GRU 
Learning Task: Classification-binary
Input Type + Scale: Text, seq_length: Small <50 and vocab_size: Large >100k 
Complexity: Wide-Deep: Number of RNN-GRU layers at least 2, embedding_dim of the Embedding layer at least min(128, upper_bound_of_vocab_size).


- Provide complete PyTorch code for the model only.
- All layer parameters must be fixed values inside the network, with no variables or external constants used.
- Avoid creating excessively large layers that would cause unreasonable memory usage.
- The model must only use standard layers directly, without defining helper functions or custom block classes.
- Output only the code, with no comments.
'''
import torch
import torch.nn as nn

class TextGRUClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(110000, 128, padding_idx=0)
        self.gru = nn.GRU(128, 160, num_layers=2, batch_first=True, bidirectional=True, dropout=0.2)
        self.dropout = nn.Dropout(0.3)
        self.fc1 = nn.Linear(320, 128)
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(128, 1)
    def forward(self, x):
        x = self.embedding(x)
        _, h_n = self.gru(x)
        h = torch.cat((h_n[-2], h_n[-1]), dim=1)
        h = self.dropout(h)
        h = self.fc1(h)
        h = self.act(h)
        h = self.dropout(h)
        out = self.fc2(h)
        return out