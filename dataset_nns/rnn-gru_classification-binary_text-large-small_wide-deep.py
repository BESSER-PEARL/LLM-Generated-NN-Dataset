'''
Generate a complete PyTorch NN code that satisfies the following four Requirements: 

Architecture: RNN-GRU 
Learning Task: Classification-binary
Input Type + Scale: Text, seq_length: Large >2000 and vocab_size: Small <1k 
Complexity: Wide-Deep: Number of RNN-GRU layers at least 2, embedding_dim of the Embedding layer at least min(128, upper_bound_of_vocab_size).


- Provide complete PyTorch code for the model only.
- All layer parameters must be fixed values inside the network, with no variables or external constants used.
- Avoid creating excessively large layers that would cause unreasonable memory usage.
- The model must only use standard layers directly, without defining helper functions or custom block classes.
- Output only the code, with no comments.
'''
import torch
import torch.nn as nn

class GRUBinaryTextClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings=900, embedding_dim=128, padding_idx=0)
        self.gru = nn.GRU(input_size=128, hidden_size=192, num_layers=2, batch_first=True, dropout=0.2, bidirectional=False)
        self.dropout = nn.Dropout(p=0.3)
        self.fc1 = nn.Linear(192, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 64)
        self.fc_out = nn.Linear(64, 1)

    def forward(self, x):
        x = self.embedding(x)
        _, h_n = self.gru(x)
        h_last = h_n[-1]
        z = self.dropout(h_last)
        z = self.fc1(z)
        z = self.relu(z)
        z = self.dropout(z)
        z = self.fc2(z)
        z = self.relu(z)
        z = self.dropout(z)
        logits = self.fc_out(z)
        return logits