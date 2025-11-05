'''
Generate a complete PyTorch NN code that satisfies the following four Requirements: 

Architecture: CNN-1D 
Learning Task: Regression
Input Type + Scale: Time series, seq_length: Large >2000 and features: Large >2000 
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
        super(CNN1DRegression, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=2048, out_channels=256, kernel_size=7, stride=2, padding=3, bias=True)
        self.bn1 = nn.BatchNorm1d(256)
        self.conv2 = nn.Conv1d(in_channels=256, out_channels=512, kernel_size=5, stride=2, padding=2, bias=True)
        self.bn2 = nn.BatchNorm1d(512)
        self.conv3 = nn.Conv1d(in_channels=512, out_channels=512, kernel_size=3, stride=2, padding=1, bias=True)
        self.bn3 = nn.BatchNorm1d(512)
        self.conv4 = nn.Conv1d(in_channels=512, out_channels=256, kernel_size=3, stride=2, padding=1, bias=True)
        self.bn4 = nn.BatchNorm1d(256)
        self.act = nn.ReLU(inplace=True)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(256, 64)
        self.dropout = nn.Dropout(p=0.1)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x):
        x = self.act(self.bn1(self.conv1(x)))
        x = self.act(self.bn2(self.conv2(x)))
        x = self.act(self.bn3(self.conv3(x)))
        x = self.act(self.bn4(self.conv4(x)))
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(self.act(self.fc1(x)))
        x = self.fc2(x)
        return x