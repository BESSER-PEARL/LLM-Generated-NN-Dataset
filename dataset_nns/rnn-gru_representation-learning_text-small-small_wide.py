'''
Generate a complete PyTorch NN code that satisfies the following four Requirements: 

Architecture: RNN-GRU 
Learning Task: Representation-learning
Input Type + Scale: Text, seq_length: Small <50 and vocab_size: Small <1k 
Complexity: Wide: embedding_dim of the Embedding layer at least min(128, upper_bound_of_vocab_size).


- Provide complete PyTorch code for the model only.
- All layer parameters must be fixed values inside the network, with no variables or external constants used.
- Avoid creating excessively large layers that would cause unreasonable memory usage.
- The model must only use standard layers directly, without defining helper functions or custom block classes.
- Output only the code, with no comments.
'''
import torch
import torch.nn as nn

class TextGRURepresentation(nn.Module):
    def __init__(self):
        super(TextGRURepresentation, self).__init__()
        self.embedding = nn.Embedding(num_embeddings=800, embedding_dim=256, padding_idx=0)
        self.gru = nn.GRU(input_size=256, hidden_size=256, num_layers=2, batch_first=True, bidirectional=True, dropout=0.1)
        self.proj = nn.Linear(512, 256)
        self.norm = nn.LayerNorm(256)
        self.activation = nn.Tanh()
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, x):
        x = self.embedding(x)
        _, h_n = self.gru(x)
        h_fwd = h_n[-2]
        h_bwd = h_n[-1]
        h = torch.cat((h_fwd, h_bwd), dim=1)
        z = self.proj(h)
        z = self.norm(z)
        z = self.activation(z)
        z = self.dropout(z)
        return z