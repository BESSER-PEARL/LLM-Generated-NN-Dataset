'''
Generate a complete PyTorch NN code that satisfies the following four Requirements: 

Architecture: CNN-1D 
Learning Task: Representation-learning
Input Type + Scale: Text, seq_length: Large >2000 and vocab_size: Large >100k 
Complexity: Deep: Number of CNN-1D layers at least 4.


- Provide complete PyTorch code for the model only.
- All layer parameters must be fixed values inside the network, with no variables or external constants used.
- Avoid creating excessively large layers that would cause unreasonable memory usage.
- The model must only use standard layers directly, without defining helper functions or custom block classes.
- Output only the code, with no comments.
'''
import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(120000, 64, padding_idx=0)
        self.conv1 = nn.Conv1d(64, 64, kernel_size=3, padding=1, dilation=1)
        self.bn1 = nn.BatchNorm1d(64)
        self.conv2 = nn.Conv1d(64, 64, kernel_size=3, padding=2, dilation=2)
        self.bn2 = nn.BatchNorm1d(64)
        self.conv3 = nn.Conv1d(64, 64, kernel_size=3, padding=4, dilation=4)
        self.bn3 = nn.BatchNorm1d(64)
        self.conv4 = nn.Conv1d(64, 64, kernel_size=3, padding=8, dilation=8)
        self.bn4 = nn.BatchNorm1d(64)
        self.conv5 = nn.Conv1d(64, 64, kernel_size=3, padding=16, dilation=16)
        self.bn5 = nn.BatchNorm1d(64)
        self.conv6 = nn.Conv1d(64, 64, kernel_size=3, padding=1, dilation=1)
        self.bn6 = nn.BatchNorm1d(64)
        self.conv7 = nn.Conv1d(64, 64, kernel_size=3, padding=2, dilation=2)
        self.bn7 = nn.BatchNorm1d(64)
        self.conv8 = nn.Conv1d(64, 64, kernel_size=3, padding=4, dilation=4)
        self.bn8 = nn.BatchNorm1d(64)
        self.dropout = nn.Dropout(0.1)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.proj = nn.Linear(64, 128)
        self.norm = nn.LayerNorm(128)

    def forward(self, x):
        x = self.embedding(x)
        x = x.transpose(1, 2)
        r = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = torch.relu(x)
        x = self.dropout(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = torch.relu(x)
        x = x + r
        x = self.dropout(x)
        r = x
        x = self.conv3(x)
        x = self.bn3(x)
        x = torch.relu(x)
        x = self.dropout(x)
        x = self.conv4(x)
        x = self.bn4(x)
        x = torch.relu(x)
        x = x + r
        x = self.dropout(x)
        r = x
        x = self.conv5(x)
        x = self.bn5(x)
        x = torch.relu(x)
        x = self.dropout(x)
        x = self.conv6(x)
        x = self.bn6(x)
        x = torch.relu(x)
        x = x + r
        x = self.dropout(x)
        r = x
        x = self.conv7(x)
        x = self.bn7(x)
        x = torch.relu(x)
        x = self.dropout(x)
        x = self.conv8(x)
        x = self.bn8(x)
        x = torch.relu(x)
        x = x + r
        x = self.pool(x).squeeze(-1)
        x = self.proj(x)
        x = self.norm(x)
        return x