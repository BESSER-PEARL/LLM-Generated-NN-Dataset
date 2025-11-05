'''
Generate a complete PyTorch NN code that satisfies the following four Requirements: 

Architecture: CNN-1D 
Learning Task: Regression
Input Type + Scale: Time series, seq_length: Small <50 and features: Large >2000 
Complexity: Wide-Deep: Number of CNN-1D layers at least 4, out_channels of first Conv layer at least min(128, upper_bound_of_features).


- Provide complete PyTorch code for the model only.
- All layer parameters must be fixed values inside the network, with no variables or external constants used.
- Avoid creating excessively large layers that would cause unreasonable memory usage.
- The model must only use standard layers directly, without defining helper functions or custom block classes.
- Output only the code, with no comments.
'''
import torch
import torch.nn as nn

class CNN1DRegressionWideDeep(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(2048, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(128)
        self.conv2 = nn.Conv1d(128, 192, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(192)
        self.conv3 = nn.Conv1d(192, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn3 = nn.BatchNorm1d(256)
        self.conv4 = nn.Conv1d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn4 = nn.BatchNorm1d(256)
        self.wide_conv = nn.Conv1d(2048, 64, kernel_size=1, stride=1, padding=0, bias=False)
        self.wide_bn = nn.BatchNorm1d(64)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.deep_fc = nn.Linear(256, 128)
        self.wide_fc = nn.Linear(64, 128)
        self.head = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1)
        )
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=0.1)
    def forward(self, x):
        d = self.conv1(x)
        d = self.bn1(d)
        d = self.relu(d)
        d = self.conv2(d)
        d = self.bn2(d)
        d = self.relu(d)
        d = self.conv3(d)
        d = self.bn3(d)
        d = self.relu(d)
        d = self.conv4(d)
        d = self.bn4(d)
        d = self.relu(d)
        d = self.pool(d)
        d = d.view(d.size(0), -1)
        d = self.relu(self.deep_fc(d))
        d = self.dropout(d)
        w = self.wide_conv(x)
        w = self.wide_bn(w)
        w = self.relu(w)
        w = self.pool(w)
        w = w.view(w.size(0), -1)
        w = self.relu(self.wide_fc(w))
        w = self.dropout(w)
        out = torch.cat([d, w], dim=1)
        out = self.head(out)
        return out