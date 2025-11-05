'''
Generate a complete PyTorch NN code that satisfies the following four Requirements: 

Architecture: RNN-GRU 
Learning Task: Classification-binary
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

class TextGRUClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings=800, embedding_dim=256, padding_idx=0)
        self.dropout_embed = nn.Dropout(p=0.2)
        self.gru = nn.GRU(input_size=256, hidden_size=128, num_layers=3, batch_first=True, bidirectional=True, dropout=0.2)
        self.dropout_fc = nn.Dropout(p=0.3)
        self.fc1 = nn.Linear(in_features=256, out_features=64)
        self.relu = nn.ReLU()
        self.fc_out = nn.Linear(in_features=64, out_features=1)

    def forward(self, x):
        x = self.embedding(x)
        x = self.dropout_embed(x)
        _, h = self.gru(x)
        h_forward = h[-2]
        h_backward = h[-1]
        h_cat = torch.cat((h_forward, h_backward), dim=1)
        x = self.fc1(h_cat)
        x = self.relu(x)
        x = self.dropout_fc(x)
        x = self.fc_out(x).squeeze(-1)
        return x