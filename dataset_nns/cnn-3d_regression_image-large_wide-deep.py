'''
Generate a complete PyTorch NN code that satisfies the following four Requirements: 

Architecture: CNN-3D 
Learning Task: Regression
Input Type + Scale: Image, resolution: Large >1024 
Complexity: Wide-Deep: Number of CNN-3D layers at least 4, out_channels of first Conv layer at least min(8, image_width//8).


- Provide complete PyTorch code for the model only.
- All layer parameters must be fixed values inside the network, with no variables or external constants used.
- Avoid creating excessively large layers that would cause unreasonable memory usage.
- The model must only use standard layers directly, without defining helper functions or custom block classes.
- Output only the code, with no comments.
'''
import torch
import torch.nn as nn

class CNN3DRegressor(nn.Module):
    def __init__(self):
        super(CNN3DRegressor, self).__init__()
        self.conv1 = nn.Conv3d(3, 16, kernel_size=(3,3,3), stride=(1,2,2), padding=(1,1,1), bias=False)
        self.bn1 = nn.BatchNorm3d(16)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(16, 32, kernel_size=(3,3,3), stride=(1,2,2), padding=(1,1,1), bias=False)
        self.bn2 = nn.BatchNorm3d(32)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv3d(32, 64, kernel_size=(3,3,3), stride=(1,2,2), padding=(1,1,1), bias=False)
        self.bn3 = nn.BatchNorm3d(64)
        self.relu3 = nn.ReLU(inplace=True)
        self.conv4 = nn.Conv3d(64, 96, kernel_size=(3,3,3), stride=(1,2,2), padding=(1,1,1), bias=False)
        self.bn4 = nn.BatchNorm3d(96)
        self.relu4 = nn.ReLU(inplace=True)
        self.conv5 = nn.Conv3d(96, 128, kernel_size=(3,3,3), stride=(1,2,2), padding=(1,1,1), bias=False)
        self.bn5 = nn.BatchNorm3d(128)
        self.relu5 = nn.ReLU(inplace=True)
        self.pool = nn.AdaptiveAvgPool3d((1,1,1))
        self.fc1 = nn.Linear(128, 64)
        self.relu_fc = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x):
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.relu3(self.bn3(self.conv3(x)))
        x = self.relu4(self.bn4(self.conv4(x)))
        x = self.relu5(self.bn5(self.conv5(x)))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.relu_fc(self.fc1(x))
        x = self.fc2(x)
        return x