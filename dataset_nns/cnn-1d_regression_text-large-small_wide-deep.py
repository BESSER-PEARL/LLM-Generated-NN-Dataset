'''
Generate a complete PyTorch NN code that satisfies the following four Requirements: 

Architecture: CNN-1D 
Learning Task: Regression
Input Type + Scale: Text, seq_length: Large >2000 and vocab_size: Small <1k 
Complexity: Wide-Deep: Number of CNN-1D layers at least 4, embedding_dim of the Embedding layer at least min(128, upper_bound_of_vocab_size).


- Provide complete PyTorch code for the model only.
- All layer parameters must be fixed values inside the network, with no variables or external constants used.
- Avoid creating excessively large layers that would cause unreasonable memory usage.
- The model must only use standard layers directly, without defining helper functions or custom block classes.
- Output only the code, with no comments.
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

class TextCNN1DRegressor(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(1000, 128, padding_idx=0)
        self.conv_a1 = nn.Conv1d(128, 128, 3, padding=1)
        self.bn_a1 = nn.BatchNorm1d(128)
        self.conv_a2 = nn.Conv1d(128, 128, 3, padding=1)
        self.bn_a2 = nn.BatchNorm1d(128)
        self.conv_b1 = nn.Conv1d(128, 128, 5, padding=2)
        self.bn_b1 = nn.BatchNorm1d(128)
        self.conv_b2 = nn.Conv1d(128, 128, 5, padding=2)
        self.bn_b2 = nn.BatchNorm1d(128)
        self.conv_c1 = nn.Conv1d(128, 128, 7, padding=3)
        self.bn_c1 = nn.BatchNorm1d(128)
        self.conv_c2 = nn.Conv1d(128, 128, 7, padding=3)
        self.bn_c2 = nn.BatchNorm1d(128)
        self.conv_d1 = nn.Conv1d(384, 256, 3, padding=1, stride=2)
        self.bn_d1 = nn.BatchNorm1d(256)
        self.conv_d2 = nn.Conv1d(256, 256, 3, padding=1)
        self.bn_d2 = nn.BatchNorm1d(256)
        self.conv_d3 = nn.Conv1d(256, 128, 3, padding=1, stride=2)
        self.bn_d3 = nn.BatchNorm1d(128)
        self.conv_d4 = nn.Conv1d(128, 128, 3, padding=1)
        self.bn_d4 = nn.BatchNorm1d(128)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.maxpool = nn.AdaptiveMaxPool1d(1)
        self.dropout = nn.Dropout(0.2)
        self.fc1 = nn.Linear(256, 64)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x):
        x = self.embedding(x)
        x = x.transpose(1, 2)
        a = F.gelu(self.bn_a1(self.conv_a1(x)))
        a = F.gelu(self.bn_a2(self.conv_a2(a)))
        b = F.gelu(self.bn_b1(self.conv_b1(x)))
        b = F.gelu(self.bn_b2(self.conv_b2(b)))
        c = F.gelu(self.bn_c1(self.conv_c1(x)))
        c = F.gelu(self.bn_c2(self.conv_c2(c)))
        y = torch.cat([a, b, c], dim=1)
        y = F.gelu(self.bn_d1(self.conv_d1(y)))
        y = F.gelu(self.bn_d2(self.conv_d2(y)))
        y = F.gelu(self.bn_d3(self.conv_d3(y)))
        y = F.gelu(self.bn_d4(self.conv_d4(y)))
        avg = self.avgpool(y).squeeze(-1)
        mx = self.maxpool(y).squeeze(-1)
        z = torch.cat([avg, mx], dim=1)
        z = self.dropout(F.gelu(self.fc1(z)))
        z = self.fc2(z)
        return z