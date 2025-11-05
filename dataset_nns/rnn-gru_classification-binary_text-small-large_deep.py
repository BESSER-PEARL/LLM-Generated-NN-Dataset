'''
Generate a complete PyTorch NN code that satisfies the following four Requirements: 

Architecture: RNN-GRU 
Learning Task: Classification-binary
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

class TextGRUClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings=100500, embedding_dim=96, padding_idx=0)
        self.gru = nn.GRU(input_size=96, hidden_size=64, num_layers=2, batch_first=True, dropout=0.2, bidirectional=False)
        self.dropout = nn.Dropout(p=0.3)
        self.fc = nn.Linear(64, 1)

    def forward(self, x):
        x = self.embedding(x)
        x, h_n = self.gru(x)
        h_last = h_n[-1]
        out = self.dropout(h_last)
        out = self.fc(out)
        return out.squeeze(-1)