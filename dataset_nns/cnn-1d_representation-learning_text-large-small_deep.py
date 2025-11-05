'''
Generate a complete PyTorch NN code that satisfies the following four Requirements: 

Architecture: CNN-1D 
Learning Task: Representation-learning
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
import torch.nn.functional as F

class TextCNN1D(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(800, 64, padding_idx=0)
        self.conv1 = nn.Conv1d(64, 128, kernel_size=7, stride=1, padding=3)
        self.bn1 = nn.BatchNorm1d(128)
        self.conv2 = nn.Conv1d(128, 128, kernel_size=7, stride=1, padding=3)
        self.bn2 = nn.BatchNorm1d(128)
        self.pool1 = nn.MaxPool1d(kernel_size=4, stride=4)
        self.conv3 = nn.Conv1d(128, 256, kernel_size=5, stride=1, padding=2)
        self.bn3 = nn.BatchNorm1d(256)
        self.conv4 = nn.Conv1d(256, 256, kernel_size=5, stride=1, padding=2)
        self.bn4 = nn.BatchNorm1d(256)
        self.pool2 = nn.MaxPool1d(kernel_size=4, stride=4)
        self.conv5 = nn.Conv1d(256, 256, kernel_size=3, stride=1, padding=1)
        self.bn5 = nn.BatchNorm1d(256)
        self.conv6 = nn.Conv1d(256, 512, kernel_size=3, stride=1, padding=1)
        self.bn6 = nn.BatchNorm1d(512)
        self.dropout = nn.Dropout(0.2)
        self.proj = nn.Linear(512, 256)
        self.out_norm = nn.LayerNorm(256)

    def forward(self, x):
        x = self.embedding(x)
        x = x.transpose(1, 2)
        x = F.gelu(self.bn1(self.conv1(x)))
        x = F.gelu(self.bn2(self.conv2(x)))
        x = self.pool1(x)
        x = F.gelu(self.bn3(self.conv3(x)))
        x = F.gelu(self.bn4(self.conv4(x)))
        x = self.pool2(x)
        x = F.gelu(self.bn5(self.conv5(x)))
        x = F.gelu(self.bn6(self.conv6(x)))
        x = self.dropout(x)
        x = F.adaptive_max_pool1d(x, 1).squeeze(-1)
        x = self.proj(x)
        x = self.out_norm(x)
        return x