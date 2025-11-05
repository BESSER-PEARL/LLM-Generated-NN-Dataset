'''
Generate a complete PyTorch NN code that satisfies the following four Requirements: 

Architecture: RNN-GRU 
Learning Task: Classification-multiclass
Input Type + Scale: Time series, seq_length: Large >2000 and features: Small <50 
Complexity: Wide: hidden_size of first RNN-GRU layer at least min(128, upper_bound_of_features).


- Provide complete PyTorch code for the model only.
- All layer parameters must be fixed values inside the network, with no variables or external constants used.
- Avoid creating excessively large layers that would cause unreasonable memory usage.
- The model must only use standard layers directly, without defining helper functions or custom block classes.
- Output only the code, with no comments.
'''
import torch
import torch.nn as nn

class RNNGRUClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.gru1 = nn.GRU(input_size=32, hidden_size=128, num_layers=1, batch_first=True, bidirectional=True)
        self.dropout1 = nn.Dropout(p=0.3)
        self.gru2 = nn.GRU(input_size=256, hidden_size=96, num_layers=1, batch_first=True, bidirectional=True)
        self.dropout2 = nn.Dropout(p=0.3)
        self.fc1 = nn.Linear(in_features=192, out_features=64)
        self.relu = nn.ReLU()
        self.fc_out = nn.Linear(in_features=64, out_features=10)

    def forward(self, x):
        out1, _ = self.gru1(x)
        out1 = self.dropout1(out1)
        _, h2 = self.gru2(out1)
        h_forward = h2[0]
        h_backward = h2[1]
        h_cat = torch.cat((h_forward, h_backward), dim=1)
        h_cat = self.dropout2(h_cat)
        z = self.fc1(h_cat)
        z = self.relu(z)
        logits = self.fc_out(z)
        return logits