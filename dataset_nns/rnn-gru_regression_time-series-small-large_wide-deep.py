'''
Generate a complete PyTorch NN code that satisfies the following four Requirements: 

Architecture: RNN-GRU 
Learning Task: Regression
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

class WideDeepGRURegressor(nn.Module):
    def __init__(self):
        super().__init__()
        self.gru1 = nn.GRU(input_size=2048, hidden_size=160, num_layers=1, batch_first=True)
        self.gru2 = nn.GRU(input_size=160, hidden_size=96, num_layers=1, batch_first=True)
        self.dropout_rnn = nn.Dropout(p=0.2)
        self.bn_rnn = nn.BatchNorm1d(96)
        self.wide_fc1 = nn.Linear(2048, 64)
        self.wide_act = nn.ReLU()
        self.wide_bn = nn.BatchNorm1d(64)
        self.combined_fc1 = nn.Linear(160, 64)
        self.combined_act1 = nn.ReLU()
        self.dropout_combined = nn.Dropout(p=0.2)
        self.output = nn.Linear(64, 1)

    def forward(self, x):
        out1, _ = self.gru1(x)
        out1 = self.dropout_rnn(out1)
        out2, _ = self.gru2(out1)
        last_deep = out2[:, -1, :]
        last_deep = self.bn_rnn(last_deep)
        x_last = x[:, -1, :]
        wide = self.wide_fc1(x_last)
        wide = self.wide_bn(wide)
        wide = self.wide_act(wide)
        comb = torch.cat([last_deep, wide], dim=1)
        comb = self.combined_fc1(comb)
        comb = self.combined_act1(comb)
        comb = self.dropout_combined(comb)
        y = self.output(comb)
        return y