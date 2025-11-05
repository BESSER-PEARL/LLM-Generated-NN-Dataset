'''
Generate a complete PyTorch NN code that satisfies the following four Requirements: 

Architecture: CNN-1D 
Learning Task: Classification-binary
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

class TextCNN1D(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(800, 64, padding_idx=0)
        self.conv1 = nn.Conv1d(64, 96, 7, padding=3)
        self.bn1 = nn.BatchNorm1d(96)
        self.pool1 = nn.MaxPool1d(2)
        self.conv2 = nn.Conv1d(96, 128, 5, padding=2)
        self.bn2 = nn.BatchNorm1d(128)
        self.pool2 = nn.MaxPool1d(2)
        self.conv3 = nn.Conv1d(128, 128, 5, padding=2)
        self.bn3 = nn.BatchNorm1d(128)
        self.conv4 = nn.Conv1d(128, 128, 3, padding=1)
        self.bn4 = nn.BatchNorm1d(128)
        self.pool3 = nn.MaxPool1d(2)
        self.conv5 = nn.Conv1d(128, 160, 3, padding=1)
        self.bn5 = nn.BatchNorm1d(160)
        self.dropout = nn.Dropout(0.2)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(160, 64)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x):
        x = self.embedding(x)
        x = x.transpose(1, 2)
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
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu(x)
        x = self.pool3(x)
        x = self.conv5(x)
        x = self.bn5(x)
        x = self.relu(x)
        x = self.global_pool(x)
        x = x.squeeze(-1)
        x = self.dropout(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x