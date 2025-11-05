'''
Generate a complete PyTorch NN code that satisfies the following four Requirements: 

Architecture: RNN-GRU 
Learning Task: Representation-learning
Input Type + Scale: Text, seq_length: Large >2000 and vocab_size: Large >100k 
Complexity: Simple: Number of RNN-GRU layers up to 2, embedding_dim of the Embedding layer up to min(128, upper_bound_of_vocab_size).


- Provide complete PyTorch code for the model only.
- All layer parameters must be fixed values inside the network, with no variables or external constants used.
- Avoid creating excessively large layers that would cause unreasonable memory usage.
- The model must only use standard layers directly, without defining helper functions or custom block classes.
- Output only the code, with no comments.
'''
import torch
import torch.nn as nn

class GRUTextRepresentation(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings=110000, embedding_dim=128, padding_idx=0)
        self.gru = nn.GRU(input_size=128, hidden_size=128, num_layers=2, batch_first=True, dropout=0.1)
        self.projection = nn.Linear(128, 64)

    def forward(self, input_ids):
        x = self.embedding(input_ids)
        _, h_n = self.gru(x)
        h = h_n[-1]
        y = self.projection(h)
        return y