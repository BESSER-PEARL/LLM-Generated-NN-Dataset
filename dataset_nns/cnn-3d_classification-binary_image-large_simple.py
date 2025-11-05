'''
Generate a complete PyTorch NN code that satisfies the following four Requirements: 

Architecture: CNN-3D 
Learning Task: Classification-binary
Input Type + Scale: Image, resolution: Large >1024 
Complexity: Simple: Number of CNN-3D layers up to 4, out_channels of first Conv layer up to min(8, image_width//8).


- Provide complete PyTorch code for the model only.
- All layer parameters must be fixed values inside the network, with no variables or external constants used.
- Avoid creating excessively large layers that would cause unreasonable memory usage.
- The model must only use standard layers directly, without defining helper functions or custom block classes.
- Output only the code, with no comments.
'''
import torch
import torch.nn as nn

class CNN3DBinaryClassifier(nn.Module):
    def __init__(self):
        super(CNN3DBinaryClassifier, self).__init__()
        self.conv1 = nn.Conv3d(1, 8, kernel_size=3, stride=(1, 2, 2), padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(8)
        self.conv2 = nn.Conv3d(8, 16, kernel_size=3, stride=(1, 2, 2), padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(16)
        self.conv3 = nn.Conv3d(16, 32, kernel_size=3, stride=(1, 2, 2), padding=1, bias=False)
        self.bn3 = nn.BatchNorm3d(32)
        self.conv4 = nn.Conv3d(32, 64, kernel_size=3, stride=(1, 2, 2), padding=1, bias=False)
        self.bn4 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.gap = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc = nn.Linear(64, 1)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.relu(self.bn4(self.conv4(x)))
        x = self.gap(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x