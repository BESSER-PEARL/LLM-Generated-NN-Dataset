'''
Generate a complete PyTorch NN code that satisfies the following four Requirements: 

Architecture: CNN-1D 
Learning Task: Representation-learning
Input Type + Scale: Time series, seq_length: Small <50 and features: Large >2000 
Complexity: Simple: Number of CNN-1D layers up to 4, out_channels of first Conv layer up to min(128, upper_bound_of_features).


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
        self.conv1 = nn.Conv1d(2048, 64, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool1d(2, 2, 0)
        self.conv2 = nn.Conv1d(64, 64, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm1d(64)
        self.pool2 = nn.MaxPool1d(2, 2, 0)
        self.conv3 = nn.Conv1d(64, 64, 3, 1, 1, bias=False)
        self.bn3 = nn.BatchNorm1d(64)
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(64, 128)
        self.norm = nn.LayerNorm(128)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = self.norm(x)
        return x