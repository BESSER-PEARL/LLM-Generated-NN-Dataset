'''
Generate a complete PyTorch NN code that satisfies the following four Requirements: 

Architecture: CNN-1D 
Learning Task: Regression
Input Type + Scale: Text, seq_length: Large >2000 and vocab_size: Small <1k 
Complexity: Deep: Number of CNN-1D layers at least 4.


- Provide complete PyTorch code for the model only.
- All layer parameters must be fixed values inside the network, with no variables or external constants used.
- Avoid creating excessively large layers that would cause unreasonable memory usage.
- The model must only use standard layers directly, without defining helper functions or custom block classes.
- Output only the code, with no comments.
'''
import torch
import torch.nn as nn

class TextCNN1DRegressor(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(800, 64, padding_idx=0)
        self.conv1 = nn.Conv1d(64, 96, kernel_size=5, stride=2, padding=2)
        self.bn1 = nn.BatchNorm1d(96)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(96, 128, kernel_size=5, stride=2, padding=2)
        self.bn2 = nn.BatchNorm1d(128)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv1d(128, 160, kernel_size=3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm1d(160)
        self.relu3 = nn.ReLU(inplace=True)
        self.conv4 = nn.Conv1d(160, 192, kernel_size=3, stride=2, padding=1)
        self.bn4 = nn.BatchNorm1d(192)
        self.relu4 = nn.ReLU(inplace=True)
        self.conv5 = nn.Conv1d(192, 224, kernel_size=3, stride=2, padding=1)
        self.bn5 = nn.BatchNorm1d(224)
        self.relu5 = nn.ReLU(inplace=True)
        self.conv6 = nn.Conv1d(224, 256, kernel_size=3, stride=1, padding=1)
        self.bn6 = nn.BatchNorm1d(256)
        self.relu6 = nn.ReLU(inplace=True)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(p=0.2)
        self.fc = nn.Linear(256, 1)

    def forward(self, x):
        x = self.embedding(x)
        x = x.transpose(1, 2)
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.relu3(self.bn3(self.conv3(x)))
        x = self.relu4(self.bn4(self.conv4(x)))
        x = self.relu5(self.bn5(self.conv5(x)))
        x = self.relu6(self.bn6(self.conv6(x)))
        x = self.pool(x).squeeze(-1)
        x = self.dropout(x)
        x = self.fc(x)
        return x