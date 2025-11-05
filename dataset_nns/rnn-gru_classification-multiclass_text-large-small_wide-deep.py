'''
Generate a complete PyTorch NN code that satisfies the following four Requirements: 

Architecture: RNN-GRU 
Learning Task: Classification-multiclass
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

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings=800, embedding_dim=256, padding_idx=0)
        self.gru = nn.GRU(input_size=256, hidden_size=192, num_layers=2, batch_first=True, dropout=0.2, bidirectional=True)
        self.dropout = nn.Dropout(0.3)
        self.fc1 = nn.Linear(384, 128)
        self.relu = nn.ReLU()
        self.fc_out = nn.Linear(128, 8)
    def forward(self, x):
        x = self.embedding(x)
        _, h_n = self.gru(x)
        h_fw = h_n[-2]
        h_bw = h_n[-1]
        h = torch.cat((h_fw, h_bw), dim=1)
        h = self.dropout(h)
        h = self.fc1(h)
        h = self.relu(h)
        h = self.dropout(h)
        logits = self.fc_out(h)
        return logits