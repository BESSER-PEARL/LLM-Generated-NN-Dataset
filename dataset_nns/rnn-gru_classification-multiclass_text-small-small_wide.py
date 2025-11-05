'''
Generate a complete PyTorch NN code that satisfies the following four Requirements: 

Architecture: RNN-GRU 
Learning Task: Classification-multiclass
Input Type + Scale: Text, seq_length: Small <50 and vocab_size: Small <1k 
Complexity: Wide: embedding_dim of the Embedding layer at least min(128, upper_bound_of_vocab_size).


- Provide complete PyTorch code for the model only.
- All layer parameters must be fixed values inside the network, with no variables or external constants used.
- Avoid creating excessively large layers that would cause unreasonable memory usage.
- The model must only use standard layers directly, without defining helper functions or custom block classes.
- Output only the code, with no comments.
'''
import torch
from torch import nn

class TextGRUClassifier(nn.Module):
    def __init__(self):
        super(TextGRUClassifier, self).__init__()
        self.embedding = nn.Embedding(num_embeddings=512, embedding_dim=128, padding_idx=0)
        self.dropout_emb = nn.Dropout(p=0.1)
        self.gru = nn.GRU(input_size=128, hidden_size=192, num_layers=2, batch_first=True, bidirectional=True, dropout=0.2)
        self.dropout_fc = nn.Dropout(p=0.2)
        self.fc = nn.Linear(384, 8)
    def forward(self, x):
        x = self.embedding(x)
        x = self.dropout_emb(x)
        _, h_n = self.gru(x)
        h_forward = h_n[-2]
        h_backward = h_n[-1]
        h = torch.cat((h_forward, h_backward), dim=1)
        h = self.dropout_fc(h)
        return self.fc(h)