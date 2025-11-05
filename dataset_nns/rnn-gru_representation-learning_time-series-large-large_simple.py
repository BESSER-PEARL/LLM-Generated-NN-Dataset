'''
Generate a complete PyTorch NN code that satisfies the following four Requirements: 

Architecture: RNN-GRU 
Learning Task: Representation-learning
Input Type + Scale: Time series, seq_length: Large >2000 and features: Large >2000 
Complexity: Simple: Number of RNN-GRU layers up to 2, hidden_size of first RNN-GRU layer up to min(128, upper_bound_of_features).


- Provide complete PyTorch code for the model only.
- All layer parameters must be fixed values inside the network, with no variables or external constants used.
- Avoid creating excessively large layers that would cause unreasonable memory usage.
- The model must only use standard layers directly, without defining helper functions or custom block classes.
- Output only the code, with no comments.
'''
import torch
import torch.nn as nn

class GRURepresentationModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.gru = nn.GRU(input_size=2048, hidden_size=128, num_layers=2, batch_first=True, dropout=0.1, bidirectional=False)
        self.embedding = nn.Linear(128, 64)
        self.activation = nn.Tanh()

    def forward(self, x):
        _, h_n = self.gru(x)
        h_last = h_n[-1]
        z = self.activation(self.embedding(h_last))
        return z