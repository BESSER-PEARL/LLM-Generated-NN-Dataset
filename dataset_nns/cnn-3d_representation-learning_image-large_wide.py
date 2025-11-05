'''
Generate a complete PyTorch NN code that satisfies the following four Requirements: 

Architecture: CNN-3D 
Learning Task: Representation-learning
Input Type + Scale: Image, resolution: Large >1024 
Complexity: Wide: out_channels of first Conv layer at least min(8, image_width//8).


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
        super().__init__()
        self.conv1 = nn.Conv3d(3, 32, kernel_size=(1,5,5), stride=(1,2,2), padding=(0,2,2), bias=False)
        self.bn1 = nn.BatchNorm3d(32)
        self.conv2 = nn.Conv3d(32, 64, kernel_size=(1,3,3), stride=(1,2,2), padding=(0,1,1), bias=False)
        self.bn2 = nn.BatchNorm3d(64)
        self.pool1 = nn.MaxPool3d(kernel_size=(1,2,2), stride=(1,2,2))
        self.conv3 = nn.Conv3d(64, 96, kernel_size=(1,3,3), stride=(1,2,2), padding=(0,1,1), bias=False)
        self.bn3 = nn.BatchNorm3d(96)
        self.conv4 = nn.Conv3d(96, 128, kernel_size=(1,3,3), stride=(1,2,2), padding=(0,1,1), bias=False)
        self.bn4 = nn.BatchNorm3d(128)
        self.conv5 = nn.Conv3d(128, 192, kernel_size=(1,3,3), stride=(1,2,2), padding=(0,1,1), bias=False)
        self.bn5 = nn.BatchNorm3d(192)
        self.gap = nn.AdaptiveAvgPool3d((1,1,1))
        self.fc = nn.Linear(192, 128)
        self.ln = nn.LayerNorm(128)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x, inplace=True)
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x, inplace=True)
        x = self.pool1(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x, inplace=True)
        x = self.conv4(x)
        x = self.bn4(x)
        x = F.relu(x, inplace=True)
        x = self.conv5(x)
        x = self.bn5(x)
        x = F.relu(x, inplace=True)
        x = self.gap(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        x = self.ln(x)
        return x