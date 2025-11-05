'''
Generate a complete PyTorch NN code that satisfies the following four Requirements: 

Architecture: CNN-1D 
Learning Task: Representation-learning
Input Type + Scale: Time series, seq_length: Large >2000 and features: Large >2000 
Complexity: Simple: Number of CNN-1D layers up to 4, out_channels of first Conv layer up to min(128, upper_bound_of_features).


- Provide complete PyTorch code for the model only.
- All layer parameters must be fixed values inside the network, with no variables or external constants used.
- Avoid creating excessively large layers that would cause unreasonable memory usage.
- The model must only use standard layers directly, without defining helper functions or custom block classes.
- Output only the code, with no comments.
'''
import torch
from torch import nn

class CNN1DRepresentationNet(nn.Module):
    def __init__(self):
        super(CNN1DRepresentationNet, self).__init__()
        self.conv1 = nn.Conv1d(2048, 64, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(64, 96, kernel_size=7, stride=4, padding=3, bias=False)
        self.bn2 = nn.BatchNorm1d(96)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv1d(96, 128, kernel_size=5, stride=4, padding=2, bias=False)
        self.bn3 = nn.BatchNorm1d(128)
        self.relu3 = nn.ReLU(inplace=True)
        self.conv4 = nn.Conv1d(128, 128, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn4 = nn.BatchNorm1d(128)
        self.relu4 = nn.ReLU(inplace=True)
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.proj = nn.Linear(128, 128, bias=True)

    def forward(self, x):
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.relu3(self.bn3(self.conv3(x)))
        x = self.relu4(self.bn4(self.conv4(x)))
        x = self.gap(x)
        x = x.squeeze(-1)
        x = self.proj(x)
        return x