'''
Generate a complete PyTorch NN code that satisfies the following four Requirements: 

Architecture: CNN-1D 
Learning Task: Regression
Input Type + Scale: Time series, seq_length: Large >2000 and features: Small <50 
Complexity: Wide: out_channels of first Conv layer at least min(128, upper_bound_of_features).


- Provide complete PyTorch code for the model only.
- All layer parameters must be fixed values inside the network, with no variables or external constants used.
- Avoid creating excessively large layers that would cause unreasonable memory usage.
- The model must only use standard layers directly, without defining helper functions or custom block classes.
- Output only the code, with no comments.
'''
import torch
import torch.nn as nn

class CNN1DRegressor(nn.Module):
    def __init__(self):
        super(CNN1DRegressor, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=32, out_channels=128, kernel_size=7, stride=2, padding=3, bias=True)
        self.bn1 = nn.BatchNorm1d(128)
        self.relu1 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool1d(kernel_size=4, stride=4)
        self.conv2 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=5, stride=1, padding=2, bias=True)
        self.bn2 = nn.BatchNorm1d(256)
        self.relu2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool1d(kernel_size=4, stride=4)
        self.conv3 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn3 = nn.BatchNorm1d(256)
        self.relu3 = nn.ReLU(inplace=True)
        self.conv4 = nn.Conv1d(in_channels=256, out_channels=128, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn4 = nn.BatchNorm1d(128)
        self.relu4 = nn.ReLU(inplace=True)
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.flatten = nn.Flatten()
        self.dropout1 = nn.Dropout(p=0.25)
        self.fc1 = nn.Linear(128, 64, bias=True)
        self.relu5 = nn.ReLU(inplace=True)
        self.dropout2 = nn.Dropout(p=0.25)
        self.fc2 = nn.Linear(64, 1, bias=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu4(x)
        x = self.gap(x)
        x = self.flatten(x)
        x = self.dropout1(x)
        x = self.fc1(x)
        x = self.relu5(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        return x