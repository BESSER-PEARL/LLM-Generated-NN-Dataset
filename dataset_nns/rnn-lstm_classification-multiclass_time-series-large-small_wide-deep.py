'''
Generate a complete PyTorch NN code that satisfies the following four Requirements: 

Architecture: RNN-LSTM 
Learning Task: Classification-multiclass
Input Type + Scale: Time series, seq_length: Large >2000 and features: Small <50 
Complexity: Wide-Deep: Number of RNN-LSTM layers at least 2, hidden_size of first RNN-LSTM layer at least min(128, upper_bound_of_features).


- Provide complete PyTorch code for the model only.
- All layer parameters must be fixed values inside the network, with no variables or external constants used.
- Avoid creating excessively large layers that would cause unreasonable memory usage.
- The model must only use standard layers directly, without defining helper functions or custom block classes.
- Output only the code, with no comments.
'''
import torch
import torch.nn as nn

class TimeSeriesLSTMClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(input_size=32, hidden_size=64, num_layers=2, batch_first=True, dropout=0.1)
        self.wide = nn.Linear(32, 32)
        self.fc1 = nn.Linear(96, 64)
        self.dropout = nn.Dropout(0.2)
        self.fc_out = nn.Linear(64, 12)

    def forward(self, x):
        outputs, (h_n, c_n) = self.lstm(x)
        deep_feat = h_n[-1]
        wide_feat = torch.relu(self.wide(x[:, -1, :]))
        z = torch.cat([deep_feat, wide_feat], dim=1)
        z = torch.relu(self.fc1(z))
        z = self.dropout(z)
        logits = self.fc_out(z)
        return logits