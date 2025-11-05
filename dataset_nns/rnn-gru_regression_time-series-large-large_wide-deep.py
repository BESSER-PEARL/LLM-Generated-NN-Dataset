'''
Generate a complete PyTorch NN code that satisfies the following four Requirements: 

Architecture: RNN-GRU 
Learning Task: Regression
Input Type + Scale: Time series, seq_length: Large >2000 and features: Large >2000 
Complexity: Wide-Deep: Number of RNN-GRU layers at least 2, hidden_size of first RNN-GRU layer at least min(128, upper_bound_of_features).


- Provide complete PyTorch code for the model only.
- All layer parameters must be fixed values inside the network, with no variables or external constants used.
- Avoid creating excessively large layers that would cause unreasonable memory usage.
- The model must only use standard layers directly, without defining helper functions or custom block classes.
- Output only the code, with no comments.
'''
import torch
import torch.nn as nn

class GRUWideDeepRegressor(nn.Module):
    def __init__(self):
        super().__init__()
        self.gru1 = nn.GRU(input_size=2048, hidden_size=256, num_layers=1, batch_first=True)
        self.dropout1 = nn.Dropout(p=0.1)
        self.gru2 = nn.GRU(input_size=256, hidden_size=128, num_layers=1, batch_first=True)
        self.dropout2 = nn.Dropout(p=0.1)
        self.fc_deep1 = nn.Linear(128, 64)
        self.relu = nn.ReLU()
        self.fc_deep_out = nn.Linear(64, 1)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc_wide1 = nn.Linear(2048, 32)
        self.fc_wide_out = nn.Linear(32, 1)

    def forward(self, x):
        out1, _ = self.gru1(x)
        out1 = self.dropout1(out1)
        _, h2 = self.gru2(out1)
        h = h2[-1]
        h = self.dropout2(h)
        deep = self.fc_deep1(h)
        deep = self.relu(deep)
        deep = self.fc_deep_out(deep)
        pooled = self.pool(x.transpose(1, 2)).squeeze(-1)
        wide = self.fc_wide1(pooled)
        wide = self.relu(wide)
        wide = self.fc_wide_out(wide)
        y = deep + wide
        return y