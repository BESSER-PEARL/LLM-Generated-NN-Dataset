'''
Generate a complete PyTorch NN code that satisfies the following four Requirements: 

Architecture: CNN-3D 
Learning Task: Representation-learning
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

class Simple3DRepresentationCNN(nn.Module):
    def __init__(self):
        super(Simple3DRepresentationCNN, self).__init__()
        self.conv1 = nn.Conv3d(3, 8, kernel_size=(3, 7, 7), stride=(1, 4, 4), padding=(1, 3, 3), bias=False)
        self.bn1 = nn.BatchNorm3d(8)
        self.pool1 = nn.MaxPool3d(kernel_size=(1, 4, 4), stride=(1, 4, 4))
        self.conv2 = nn.Conv3d(8, 12, kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=(1, 1, 1), bias=False)
        self.bn2 = nn.BatchNorm3d(12)
        self.pool2 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))
        self.conv3 = nn.Conv3d(12, 16, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=False)
        self.bn3 = nn.BatchNorm3d(16)
        self.act = nn.ReLU(inplace=True)
        self.global_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.flatten = nn.Flatten(start_dim=1)
        self.dropout = nn.Dropout(p=0.1)
        self.fc = nn.Linear(16, 64)
        self.norm = nn.LayerNorm(64)
        self.out = nn.Tanh()

    def forward(self, x):
        x = self.act(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        x = self.act(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        x = self.act(self.bn3(self.conv3(x)))
        x = self.global_pool(x)
        x = self.flatten(x)
        x = self.dropout(x)
        x = self.fc(x)
        x = self.norm(x)
        x = self.out(x)
        return x