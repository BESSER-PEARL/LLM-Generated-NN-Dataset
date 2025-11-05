'''
Generate a complete PyTorch NN code that satisfies the following four Requirements: 

Architecture: CNN-1D 
Learning Task: Classification-binary
Input Type + Scale: Time series, seq_length: Small <50 and features: Small <50 
Complexity: Wide-Deep: Number of CNN-1D layers at least 4, out_channels of first Conv layer at least min(128, upper_bound_of_features).


- Provide complete PyTorch code for the model only.
- All layer parameters must be fixed values inside the network, with no variables or external constants used.
- Avoid creating excessively large layers that would cause unreasonable memory usage.
- The model must only use standard layers directly, without defining helper functions or custom block classes.
- Output only the code, with no comments.
'''
import torch
import torch.nn as nn

class CNN1DWideDeepBinary(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(32, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(128)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(128, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(128)
        self.relu2 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv1d(128, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn3 = nn.BatchNorm1d(256)
        self.relu3 = nn.ReLU(inplace=True)
        self.conv4 = nn.Conv1d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn4 = nn.BatchNorm1d(256)
        self.relu4 = nn.ReLU(inplace=True)
        self.gap_deep = nn.AdaptiveAvgPool1d(1)
        self.drop_deep = nn.Dropout(p=0.2)
        self.convw1 = nn.Conv1d(32, 64, kernel_size=7, stride=1, padding=3, bias=True)
        self.reluw1 = nn.ReLU(inplace=True)
        self.convw2 = nn.Conv1d(64, 64, kernel_size=5, stride=1, padding=4, dilation=2, bias=True)
        self.reluw2 = nn.ReLU(inplace=True)
        self.gap_wide = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(320, 128, bias=True)
        self.relu_fc1 = nn.ReLU(inplace=True)
        self.drop_fc = nn.Dropout(p=0.3)
        self.fc2 = nn.Linear(128, 1, bias=True)

    def forward(self, x):
        d = self.conv1(x)
        d = self.bn1(d)
        d = self.relu1(d)
        d = self.conv2(d)
        d = self.bn2(d)
        d = self.relu2(d)
        d = self.pool1(d)
        d = self.conv3(d)
        d = self.bn3(d)
        d = self.relu3(d)
        d = self.conv4(d)
        d = self.bn4(d)
        d = self.relu4(d)
        d = self.gap_deep(d)
        d = self.drop_deep(d)
        d = torch.flatten(d, 1)
        w = self.convw1(x)
        w = self.reluw1(w)
        w = self.convw2(w)
        w = self.reluw2(w)
        w = self.gap_wide(w)
        w = torch.flatten(w, 1)
        z = torch.cat([d, w], dim=1)
        z = self.fc1(z)
        z = self.relu_fc1(z)
        z = self.drop_fc(z)
        z = self.fc2(z)
        return z