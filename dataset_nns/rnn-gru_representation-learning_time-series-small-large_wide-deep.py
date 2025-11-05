'''
Generate a complete PyTorch NN code that satisfies the following four Requirements: 

Architecture: RNN-GRU 
Learning Task: Representation-learning
Input Type + Scale: Time series, seq_length: Small <50 and features: Large >2000 
Complexity: Wide-Deep: Number of RNN-GRU layers at least 2, hidden_size of first RNN-GRU layer at least min(128, upper_bound_of_features).


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
        self.gru1 = nn.GRU(input_size=2048, hidden_size=192, num_layers=1, batch_first=True)
        self.dropout1 = nn.Dropout(p=0.2)
        self.gru2 = nn.GRU(input_size=192, hidden_size=128, num_layers=1, batch_first=True)
        self.dropout2 = nn.Dropout(p=0.2)
        self.act = nn.GELU()
        self.fc = nn.Linear(256, 128)
        self.norm = nn.LayerNorm(128)

    def forward(self, x):
        out1, _ = self.gru1(x)
        out1 = self.dropout1(out1)
        out2, _ = self.gru2(out1)
        out2 = self.dropout2(out2)
        last = out2[:, -1, :]
        mean = out2.mean(dim=1)
        rep = torch.cat([last, mean], dim=1)
        rep = self.fc(rep)
        rep = self.act(rep)
        rep = self.norm(rep)
        return rep