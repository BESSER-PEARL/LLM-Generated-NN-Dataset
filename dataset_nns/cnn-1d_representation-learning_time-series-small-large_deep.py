'''
Generate a complete PyTorch NN code that satisfies the following four Requirements: 

Architecture: CNN-1D 
Learning Task: Representation-learning
Input Type + Scale: Time series, seq_length: Small <50 and features: Large >2000 
Complexity: Deep: Number of CNN-1D layers at least 4.


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
        self.conv1 = nn.Conv1d(2048, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(256)
        self.conv2 = nn.Conv1d(256, 256, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(256)
        self.conv3 = nn.Conv1d(256, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn3 = nn.BatchNorm1d(128)
        self.conv4 = nn.Conv1d(128, 128, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn4 = nn.BatchNorm1d(128)
        self.conv5 = nn.Conv1d(128, 128, kernel_size=3, stride=1, padding=2, dilation=2, bias=False)
        self.bn5 = nn.BatchNorm1d(128)
        self.conv6 = nn.Conv1d(128, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn6 = nn.BatchNorm1d(64)
        self.conv7 = nn.Conv1d(64, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn7 = nn.BatchNorm1d(64)
        self.conv8 = nn.Conv1d(64, 128, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn8 = nn.BatchNorm1d(128)
        self.act = nn.GELU()
        self.drop1 = nn.Dropout(p=0.1)
        self.drop2 = nn.Dropout(p=0.1)
        self.drop3 = nn.Dropout(p=0.1)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.proj_fc1 = nn.Linear(128, 128, bias=False)
        self.proj_bn1 = nn.BatchNorm1d(128)
        self.proj_drop = nn.Dropout(p=0.2)
        self.proj_fc2 = nn.Linear(128, 64, bias=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.act(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.act(x)
        x = self.drop2(x)
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.act(x)
        x = self.conv5(x)
        x = self.bn5(x)
        x = self.act(x)
        x = self.drop3(x)
        x = self.conv6(x)
        x = self.bn6(x)
        x = self.act(x)
        x = self.conv7(x)
        x = self.bn7(x)
        x = self.act(x)
        x = self.conv8(x)
        x = self.bn8(x)
        x = self.act(x)
        x = self.pool(x)
        x = x.flatten(1)
        x = self.proj_fc1(x)
        x = self.proj_bn1(x)
        x = self.act(x)
        x = self.proj_drop(x)
        x = self.proj_fc2(x)
        return x