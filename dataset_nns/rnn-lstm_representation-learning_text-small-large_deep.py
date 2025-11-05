'''
Generate a complete PyTorch NN code that satisfies the following four Requirements: 

Architecture: RNN-LSTM 
Learning Task: Representation-learning
Input Type + Scale: Text, seq_length: Small <50 and vocab_size: Large >100k 
Complexity: Deep: Number of RNN-LSTM layers at least 2.


- Provide complete PyTorch code for the model only.
- All layer parameters must be fixed values inside the network, with no variables or external constants used.
- Avoid creating excessively large layers that would cause unreasonable memory usage.
- The model must only use standard layers directly, without defining helper functions or custom block classes.
- Output only the code, with no comments.
'''
import torch
import torch.nn as nn

class RNNLSTMRepresentation(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings=100256, embedding_dim=64, padding_idx=0)
        self.dropout = nn.Dropout(p=0.1)
        self.lstm = nn.LSTM(input_size=64, hidden_size=256, num_layers=3, batch_first=True, dropout=0.2, bidirectional=False)
        self.layer_norm = nn.LayerNorm(256)
        self.proj = nn.Linear(256, 128, bias=True)
        self.out_norm = nn.LayerNorm(128)

    def forward(self, x):
        x = self.embedding(x)
        x = self.dropout(x)
        x, (h, _) = self.lstm(x)
        x = h[-1]
        x = self.layer_norm(x)
        x = torch.tanh(self.proj(x))
        x = self.out_norm(x)
        return x