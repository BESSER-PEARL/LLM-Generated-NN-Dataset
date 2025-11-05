'''
Generate a complete PyTorch NN code that satisfies the following four Requirements: 

Architecture: RNN-LSTM 
Learning Task: Classification-multiclass
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

class WideDeepLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(input_size=2048, hidden_size=256, num_layers=2, batch_first=True, dropout=0.1)
        self.deep_fc = nn.Linear(256, 128)
        self.deep_act = nn.ReLU()
        self.deep_drop = nn.Dropout(0.2)
        self.wide_fc = nn.Linear(2048, 128)
        self.wide_act = nn.ReLU()
        self.wide_drop = nn.Dropout(0.2)
        self.classifier = nn.Linear(256, 10)

    def forward(self, x):
        out, (h_n, c_n) = self.lstm(x)
        deep_feat = h_n[-1]
        deep_feat = self.deep_fc(deep_feat)
        deep_feat = self.deep_act(deep_feat)
        deep_feat = self.deep_drop(deep_feat)
        wide_feat = self.wide_fc(x)
        wide_feat = self.wide_act(wide_feat)
        wide_feat = torch.mean(wide_feat, dim=1)
        wide_feat = self.wide_drop(wide_feat)
        features = torch.cat([deep_feat, wide_feat], dim=1)
        logits = self.classifier(features)
        return logits