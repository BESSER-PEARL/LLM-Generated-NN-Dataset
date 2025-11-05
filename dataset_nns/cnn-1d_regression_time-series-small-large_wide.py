'''
Generate a complete PyTorch NN code that satisfies the following four Requirements: 

Architecture: CNN-1D 
Learning Task: Regression
Input Type + Scale: Time series, seq_length: Small <50 and features: Large >2000 
Complexity: Wide: out_channels of first Conv layer at least min(128, upper_bound_of_features).


- Provide complete PyTorch code for the model only.
- All layer parameters must be fixed values inside the network, with no variables or external constants used.
- Avoid creating excessively large layers that would cause unreasonable memory usage.
- The model must only use standard layers directly, without defining helper functions or custom block classes.
- Output only the code, with no comments.
'''
import torch
import torch.nn as nn

class CNN1DRegression(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(2048, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(256)
        self.act1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(256)
        self.act2 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2, padding=0)
        self.conv3 = nn.Conv1d(256, 192, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn3 = nn.BatchNorm1d(192)
        self.act3 = nn.ReLU(inplace=True)
        self.conv4 = nn.Conv1d(192, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn4 = nn.BatchNorm1d(128)
        self.act4 = nn.ReLU(inplace=True)
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(p=0.1)
        self.fc1 = nn.Linear(128, 64)
        self.act5 = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x):
        x = self.act1(self.bn1(self.conv1(x)))
        x = self.pool1(self.act2(self.bn2(self.conv2(x))))
        x = self.act3(self.bn3(self.conv3(x)))
        x = self.act4(self.bn4(self.conv4(x)))
        x = self.gap(x)
        x = torch.flatten(x, 1)
        x = self.dropout(self.act5(self.fc1(x)))
        x = self.fc2(x)
        return x