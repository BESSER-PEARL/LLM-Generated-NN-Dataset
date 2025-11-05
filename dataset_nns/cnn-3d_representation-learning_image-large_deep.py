'''
Generate a complete PyTorch NN code that satisfies the following four Requirements: 

Architecture: CNN-3D 
Learning Task: Representation-learning
Input Type + Scale: Image, resolution: Large >1024 
Complexity: Deep: Number of CNN-3D layers at least 4.


- Provide complete PyTorch code for the model only.
- All layer parameters must be fixed values inside the network, with no variables or external constants used.
- Avoid creating excessively large layers that would cause unreasonable memory usage.
- The model must only use standard layers directly, without defining helper functions or custom block classes.
- Output only the code, with no comments.
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN3DRepresentation(nn.Module):
    def __init__(self):
        super(CNN3DRepresentation, self).__init__()
        self.conv1 = nn.Conv3d(3, 16, kernel_size=3, stride=(1, 2, 2), padding=1, bias=True)
        self.bn1 = nn.BatchNorm3d(16)
        self.conv2 = nn.Conv3d(16, 24, kernel_size=3, stride=(1, 2, 2), padding=1, bias=True)
        self.bn2 = nn.BatchNorm3d(24)
        self.conv3 = nn.Conv3d(24, 32, kernel_size=3, stride=(1, 2, 2), padding=1, bias=True)
        self.bn3 = nn.BatchNorm3d(32)
        self.conv4 = nn.Conv3d(32, 48, kernel_size=3, stride=(1, 2, 2), padding=1, bias=True)
        self.bn4 = nn.BatchNorm3d(48)
        self.conv5 = nn.Conv3d(48, 64, kernel_size=3, stride=(1, 1, 1), padding=1, bias=True)
        self.bn5 = nn.BatchNorm3d(64)
        self.conv6 = nn.Conv3d(64, 96, kernel_size=3, stride=(1, 1, 1), padding=1, bias=True)
        self.bn6 = nn.BatchNorm3d(96)
        self.conv7 = nn.Conv3d(96, 128, kernel_size=3, stride=(1, 1, 1), padding=1, bias=True)
        self.bn7 = nn.BatchNorm3d(128)
        self.conv8 = nn.Conv3d(128, 128, kernel_size=3, stride=(1, 1, 1), padding=1, bias=True)
        self.bn8 = nn.BatchNorm3d(128)
        self.pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc1 = nn.Linear(128, 256, bias=True)
        self.dropout = nn.Dropout(p=0.1)
        self.fc2 = nn.Linear(256, 128, bias=True)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)), inplace=True)
        x = F.relu(self.bn2(self.conv2(x)), inplace=True)
        x = F.relu(self.bn3(self.conv3(x)), inplace=True)
        x = F.relu(self.bn4(self.conv4(x)), inplace=True)
        x = F.relu(self.bn5(self.conv5(x)), inplace=True)
        x = F.relu(self.bn6(self.conv6(x)), inplace=True)
        x = F.relu(self.bn7(self.conv7(x)), inplace=True)
        x = F.relu(self.bn8(self.conv8(x)), inplace=True)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x), inplace=True)
        x = self.dropout(x)
        x = self.fc2(x)
        x = F.normalize(x, p=2, dim=1)
        return x