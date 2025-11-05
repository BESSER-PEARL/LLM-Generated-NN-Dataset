'''
Generate a complete PyTorch NN code that satisfies the following four Requirements: 

Architecture: RNN-GRU 
Learning Task: Classification-binary
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

class BinaryTextGRU(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings=800, embedding_dim=128, padding_idx=0)
        self.gru = nn.GRU(input_size=128, hidden_size=128, num_layers=1, batch_first=True, bidirectional=False)
        self.dropout = nn.Dropout(p=0.2)
        self.fc = nn.Linear(128, 1)

    def forward(self, x):
        x = self.embedding(x)
        _, h_n = self.gru(x)
        h = h_n[-1]
        h = self.dropout(h)
        logits = self.fc(h)
        return logits