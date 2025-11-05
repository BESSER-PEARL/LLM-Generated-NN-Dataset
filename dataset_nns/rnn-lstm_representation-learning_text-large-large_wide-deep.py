'''
Generate a complete PyTorch NN code that satisfies the following four Requirements: 

Architecture: RNN-LSTM 
Learning Task: Representation-learning
Input Type + Scale: Text, seq_length: Large >2000 and vocab_size: Large >100k 
Complexity: Wide-Deep: Number of RNN-LSTM layers at least 2, embedding_dim of the Embedding layer at least min(128, upper_bound_of_vocab_size).


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
        self.dropout = nn.Dropout(p=0.2)
        self.lstm = nn.LSTM(input_size=128, hidden_size=256, num_layers=2, batch_first=True, bidirectional=True, dropout=0.2)
        self.projection = nn.Linear(in_features=512, out_features=256)
        self.norm = nn.LayerNorm(normalized_shape=256)
        self.activation = nn.Tanh()

    def forward(self, x):
        x = self.embedding(x)
        x = self.dropout(x)
        _, (h_n, _) = self.lstm(x)
        h_forward = h_n[-2]
        h_backward = h_n[-1]
        h = torch.cat((h_forward, h_backward), dim=1)
        z = self.projection(h)
        z = self.norm(z)
        z = self.activation(z)
        return z