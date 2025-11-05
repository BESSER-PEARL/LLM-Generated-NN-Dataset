'''
Generate a complete PyTorch NN code that satisfies the following four Requirements: 

Architecture: RNN-GRU 
Learning Task: Classification-binary
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

class RNNGRUWideDeepBinaryClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.gru = nn.GRU(input_size=2048, hidden_size=256, num_layers=2, batch_first=True, bidirectional=True, dropout=0.1)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.wide_fc = nn.Linear(2048, 128)
        self.wide_bn = nn.BatchNorm1d(128)
        self.wide_act = nn.ReLU()
        self.deep_fc = nn.Linear(512, 128)
        self.deep_bn = nn.BatchNorm1d(128)
        self.deep_act = nn.ReLU()
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        _, h_n = self.gru(x)
        deep_feat = torch.cat((h_n[-2], h_n[-1]), dim=1)
        deep_feat = self.deep_act(self.deep_bn(self.deep_fc(deep_feat)))
        wide_input = x.transpose(1, 2)
        wide_vec = self.pool(wide_input).squeeze(-1)
        wide_feat = self.wide_act(self.wide_bn(self.wide_fc(wide_vec)))
        fused = torch.cat((deep_feat, wide_feat), dim=1)
        out = self.classifier(fused)
        return out