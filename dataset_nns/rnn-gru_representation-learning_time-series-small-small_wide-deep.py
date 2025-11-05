'''
Generate a complete PyTorch NN code that satisfies the following four Requirements: 

Architecture: RNN-GRU 
Learning Task: Representation-learning
Input Type + Scale: Time series, seq_length: Small <50 and features: Small <50 
Complexity: Wide-Deep: Number of RNN-GRU layers at least 2, hidden_size of first RNN-GRU layer at least min(128, upper_bound_of_features).


- Provide complete PyTorch code for the model only.
- All layer parameters must be fixed values inside the network, with no variables or external constants used.
- Avoid creating excessively large layers that would cause unreasonable memory usage.
- The model must only use standard layers directly, without defining helper functions or custom block classes.
- Output only the code, with no comments.
'''
import torch
import torch.nn as nn

class WideDeepGRU(nn.Module):
    def __init__(self):
        super().__init__()
        self.gru = nn.GRU(input_size=32, hidden_size=128, num_layers=2, batch_first=True, dropout=0.1)
        self.layer_norm_deep = nn.LayerNorm(128)
        self.proj_deep = nn.Linear(128, 64)
        self.proj_wide = nn.Linear(32, 64)
        self.activation = nn.Tanh()
        self.dropout = nn.Dropout(p=0.1)
        self.layer_norm_out = nn.LayerNorm(64)

    def forward(self, x):
        _, h_n = self.gru(x)
        deep = h_n[-1]
        deep = self.layer_norm_deep(deep)
        deep = self.proj_deep(deep)
        deep = self.activation(deep)
        deep = self.dropout(deep)
        wide = x.mean(dim=1)
        wide = self.proj_wide(wide)
        out = deep + wide
        out = self.layer_norm_out(out)
        return out