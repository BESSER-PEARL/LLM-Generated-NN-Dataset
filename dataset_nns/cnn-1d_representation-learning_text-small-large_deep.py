'''
Generate a complete PyTorch NN code that satisfies the following four Requirements: 

Architecture: CNN-1D 
Learning Task: Representation-learning
Input Type + Scale: Text, seq_length: Small <50 and vocab_size: Large >100k 
Complexity: Deep: Number of CNN-1D layers at least 4.


- Provide complete PyTorch code for the model only.
- All layer parameters must be fixed values inside the network, with no variables or external constants used.
- Avoid creating excessively large layers that would cause unreasonable memory usage.
- The model must only use standard layers directly, without defining helper functions or custom block classes.
- Output only the code, with no comments.
'''
import torch
import torch.nn as nn

class TextCNN1DEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(120000, 96, padding_idx=0)
        self.conv1 = nn.Conv1d(96, 128, 3, stride=1, padding=1, dilation=1, groups=1, bias=True)
        self.bn1 = nn.BatchNorm1d(128, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2 = nn.Conv1d(128, 128, 5, stride=1, padding=2, dilation=1, groups=1, bias=True)
        self.bn2 = nn.BatchNorm1d(128, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True)
        self.conv3 = nn.Conv1d(128, 128, 3, stride=1, padding=2, dilation=2, groups=1, bias=True)
        self.bn3 = nn.BatchNorm1d(128, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True)
        self.conv4 = nn.Conv1d(128, 128, 7, stride=1, padding=3, dilation=1, groups=1, bias=True)
        self.bn4 = nn.BatchNorm1d(128, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True)
        self.conv5 = nn.Conv1d(128, 192, 3, stride=1, padding=1, dilation=1, groups=1, bias=True)
        self.bn5 = nn.BatchNorm1d(192, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True)
        self.conv6 = nn.Conv1d(192, 256, 3, stride=1, padding=1, dilation=1, groups=1, bias=True)
        self.bn6 = nn.BatchNorm1d(256, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True)
        self.dropout = nn.Dropout(p=0.2, inplace=False)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.proj = nn.Linear(256, 256, bias=True)
        self.act = nn.ReLU(inplace=False)
        self.norm = nn.LayerNorm(256, eps=1e-5, elementwise_affine=True)
    def forward(self, x):
        x = self.embedding(x)
        x = self.dropout(x)
        x = x.transpose(1, 2)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.act(x)
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.conv5(x)
        x = self.bn5(x)
        x = self.act(x)
        x = self.conv6(x)
        x = self.bn6(x)
        x = self.act(x)
        x = self.pool(x)
        x = x.squeeze(-1)
        x = self.proj(x)
        x = self.act(x)
        x = self.norm(x)
        return x