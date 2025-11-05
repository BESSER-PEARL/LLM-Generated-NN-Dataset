'''
Generate a complete PyTorch NN code that satisfies the following four Requirements: 

Architecture: RNN-LSTM 
Learning Task: Classification-binary
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
        self.lstm1 = nn.LSTM(input_size=32, hidden_size=128, num_layers=1, batch_first=True)
        self.lstm2 = nn.LSTM(input_size=128, hidden_size=64, num_layers=1, batch_first=True)
        self.wide_fc = nn.Sequential(
            nn.Linear(32, 32),
            nn.ReLU()
        )
        self.classifier = nn.Sequential(
            nn.Linear(96, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        out1, _ = self.lstm1(x)
        out2, _ = self.lstm2(out1)
        deep_feat = out2[:, -1, :]
        wide_feat = x.mean(dim=1)
        wide_feat = self.wide_fc(wide_feat)
        combined = torch.cat([deep_feat, wide_feat], dim=1)
        logits = self.classifier(combined)
        return logits