'''
Generate a complete PyTorch NN code that satisfies the following four Requirements: 

Architecture: RNN-LSTM 
Learning Task: Representation-learning
Input Type + Scale: Text, seq_length: Small <50 and vocab_size: Large >100k 
Complexity: Wide: embedding_dim of the Embedding layer at least min(128, upper_bound_of_vocab_size).


- Provide complete PyTorch code for the model only.
- All layer parameters must be fixed values inside the network, with no variables or external constants used.
- Avoid creating excessively large layers that would cause unreasonable memory usage.
- The model must only use standard layers directly, without defining helper functions or custom block classes.
- Output only the code, with no comments.
'''
import torch
import torch.nn as nn

class TextLSTMRepresentation(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings=120000, embedding_dim=128, padding_idx=0)
        self.dropout_in = nn.Dropout(p=0.1)
        self.lstm = nn.LSTM(input_size=128, hidden_size=256, num_layers=2, batch_first=True, dropout=0.2, bidirectional=True)
        self.norm = nn.LayerNorm(512)
        self.dropout_out = nn.Dropout(p=0.1)
        self.proj = nn.Linear(512, 256)
        self.act = nn.Tanh()

    def forward(self, x):
        x = self.embedding(x)
        x = self.dropout_in(x)
        _, (h, _) = self.lstm(x)
        h_cat = torch.cat((h[-2], h[-1]), dim=1)
        h_cat = self.norm(h_cat)
        h_cat = self.dropout_out(h_cat)
        z = self.proj(h_cat)
        z = self.act(z)
        return z