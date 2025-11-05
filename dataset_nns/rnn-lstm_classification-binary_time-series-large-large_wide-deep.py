'''
Generate a complete PyTorch NN code that satisfies the following four Requirements: 

Architecture: RNN-LSTM 
Learning Task: Classification-binary
Input Type + Scale: Time series, seq_length: Large >2000 and features: Large >2000 
Complexity: Wide-Deep: Number of RNN-LSTM layers at least 2, hidden_size of first RNN-LSTM layer at least min(128, upper_bound_of_features).


- Provide complete PyTorch code for the model only.
- All layer parameters must be fixed values inside the network, with no variables or external constants used.
- Avoid creating excessively large layers that would cause unreasonable memory usage.
- The model must only use standard layers directly, without defining helper functions or custom block classes.
- Output only the code, with no comments.
'''
import torch
import torch.nn as nn

class WideDeepLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(input_size=2048, hidden_size=128, num_layers=2, batch_first=True, dropout=0.1, bidirectional=False)
        self.wide_fc = nn.Linear(2048, 64)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.2)
        self.classifier_fc1 = nn.Linear(192, 64)
        self.classifier_fc2 = nn.Linear(64, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        deep_feat = out[:, -1, :]
        wide_feat = self.relu(self.wide_fc(x[:, -1, :]))
        combined = torch.cat([deep_feat, wide_feat], dim=1)
        h = self.relu(self.classifier_fc1(combined))
        h = self.dropout(h)
        logit = self.classifier_fc2(h)
        return logit