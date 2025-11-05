'''
Generate a complete PyTorch NN code that satisfies the following four Requirements: 

Architecture: CNN-1D 
Learning Task: Representation-learning
Input Type + Scale: Time series, seq_length: Small <50 and features: Small <50 
Complexity: Wide: out_channels of first Conv layer at least min(128, upper_bound_of_features).


- Provide complete PyTorch code for the model only.
- All layer parameters must be fixed values inside the network, with no variables or external constants used.
- Avoid creating excessively large layers that would cause unreasonable memory usage.
- The model must only use standard layers directly, without defining helper functions or custom block classes.
- Output only the code, with no comments.
'''
import torch
import torch.nn as nn

class CNN1DRepresentation(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(49, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(128)
        self.act1 = nn.GELU()
        self.drop1 = nn.Dropout(p=0.1)
        self.conv2 = nn.Conv1d(128, 192, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(192)
        self.act2 = nn.GELU()
        self.drop2 = nn.Dropout(p=0.1)
        self.conv3 = nn.Conv1d(192, 256, kernel_size=3, stride=1, padding=2, dilation=2, bias=False)
        self.bn3 = nn.BatchNorm1d(256)
        self.act3 = nn.GELU()
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(256, 128, bias=False)
        self.bn4 = nn.BatchNorm1d(128)
        self.act4 = nn.GELU()
        self.fc2 = nn.Linear(128, 64, bias=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.drop1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.act2(x)
        x = self.drop2(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.act3(x)
        x = self.pool(x)
        x = x.squeeze(-1)
        x = self.fc1(x)
        x = self.bn4(x)
        x = self.act4(x)
        x = self.fc2(x)
        return x