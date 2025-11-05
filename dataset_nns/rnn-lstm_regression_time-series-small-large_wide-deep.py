'''
Generate a complete PyTorch NN code that satisfies the following four Requirements: 

Architecture: RNN-LSTM 
Learning Task: Regression
Input Type + Scale: Time series, seq_length: Small <50 and features: Large >2000 
Complexity: Wide-Deep: Number of RNN-LSTM layers at least 2, hidden_size of first RNN-LSTM layer at least min(128, upper_bound_of_features).


- Provide complete PyTorch code for the model only.
- All layer parameters must be fixed values inside the network, with no variables or external constants used.
- Avoid creating excessively large layers that would cause unreasonable memory usage.
- The model must only use standard layers directly, without defining helper functions or custom block classes.
- Output only the code, with no comments.
'''
import torch
import torch.nn as nn

class TimeSeriesLSTMRegressor(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm1 = nn.LSTM(input_size=2048, hidden_size=256, num_layers=1, batch_first=True, bidirectional=False)
        self.lstm2 = nn.LSTM(input_size=256, hidden_size=128, num_layers=1, batch_first=True, bidirectional=False)
        self.lstm3 = nn.LSTM(input_size=128, hidden_size=64, num_layers=1, batch_first=True, bidirectional=False)
        self.dropout = nn.Dropout(p=0.2)
        self.deep_fc = nn.Linear(64, 32)
        self.deep_act = nn.ReLU()
        self.wide_fc = nn.Linear(2048, 32)
        self.wide_act = nn.ReLU()
        self.out_fc = nn.Linear(64, 1)
    def forward(self, x):
        out, _ = self.lstm1(x)
        out, _ = self.lstm2(out)
        out, _ = self.lstm3(out)
        deep = out[:, -1, :]
        deep = self.dropout(deep)
        deep = self.deep_fc(deep)
        deep = self.deep_act(deep)
        wide = x[:, -1, :]
        wide = self.wide_fc(wide)
        wide = self.wide_act(wide)
        comb = torch.cat([deep, wide], dim=1)
        comb = self.dropout(comb)
        y = self.out_fc(comb)
        return y