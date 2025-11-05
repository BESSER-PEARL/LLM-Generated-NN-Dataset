'''
Generate a complete PyTorch NN code that satisfies the following four Requirements: 

Architecture: CNN-3D 
Learning Task: Classification-multiclass
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

class CNN3DClassifierLarge(nn.Module):
    def __init__(self):
        super(CNN3DClassifierLarge, self).__init__()
        self.conv1 = nn.Conv3d(3, 32, kernel_size=(3,7,7), stride=(1,4,4), padding=(1,3,3), bias=False)
        self.bn1 = nn.BatchNorm3d(32)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(32, 64, kernel_size=3, stride=(1,2,2), padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(64)
        self.relu2 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool3d(kernel_size=(1,2,2), stride=(1,2,2))
        self.conv3 = nn.Conv3d(64, 128, kernel_size=3, stride=(1,2,2), padding=1, bias=False)
        self.bn3 = nn.BatchNorm3d(128)
        self.relu3 = nn.ReLU(inplace=True)
        self.conv4 = nn.Conv3d(128, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn4 = nn.BatchNorm3d(128)
        self.relu4 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool3d(kernel_size=(1,2,2), stride=(1,2,2))
        self.conv5 = nn.Conv3d(128, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn5 = nn.BatchNorm3d(256)
        self.relu5 = nn.ReLU(inplace=True)
        self.conv6 = nn.Conv3d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn6 = nn.BatchNorm3d(256)
        self.relu6 = nn.ReLU(inplace=True)
        self.conv7 = nn.Conv3d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn7 = nn.BatchNorm3d(256)
        self.relu7 = nn.ReLU(inplace=True)
        self.conv8 = nn.Conv3d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn8 = nn.BatchNorm3d(256)
        self.relu8 = nn.ReLU(inplace=True)
        self.gap = nn.AdaptiveAvgPool3d((1,1,1))
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(256, 128)
        self.relu9 = nn.ReLU(inplace=True)
        self.drop = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.pool1(x)
        x = self.relu3(self.bn3(self.conv3(x)))
        x = self.relu4(self.bn4(self.conv4(x)))
        x = self.pool2(x)
        x = self.relu5(self.bn5(self.conv5(x)))
        x = self.relu6(self.bn6(self.conv6(x)))
        x = self.relu7(self.bn7(self.conv7(x)))
        x = self.relu8(self.bn8(self.conv8(x)))
        x = self.gap(x)
        x = self.flatten(x)
        x = self.relu9(self.fc1(x))
        x = self.drop(x)
        x = self.fc2(x)
        return x