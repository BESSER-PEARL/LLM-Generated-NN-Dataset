'''
Generate a complete PyTorch NN code that satisfies the following four Requirements: 

Architecture: CNN-1D 
Learning Task: Classification-multiclass
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
import torch.nn.functional as F

class TextCNN1D(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(120000, 128, padding_idx=0)
        self.embed_dropout = nn.Dropout(0.1)
        self.conv1 = nn.Conv1d(128, 128, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(128)
        self.conv2 = nn.Conv1d(128, 128, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(128)
        self.conv3 = nn.Conv1d(128, 128, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm1d(128)
        self.conv4 = nn.Conv1d(128, 128, kernel_size=3, padding=1, bias=False)
        self.bn4 = nn.BatchNorm1d(128)
        self.conv5 = nn.Conv1d(128, 128, kernel_size=3, padding=1, bias=False)
        self.bn5 = nn.BatchNorm1d(128)
        self.dropout = nn.Dropout(0.2)
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.fc = nn.Linear(128, 20)
    def forward(self, x):
        x = self.embedding(x)
        x = self.embed_dropout(x)
        x = x.permute(0, 2, 1)
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.conv4(x)
        x = self.bn4(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.conv5(x)
        x = self.bn5(x)
        x = F.relu(x)
        x = self.pool(x)
        x = x.squeeze(-1)
        x = self.fc(x)
        return x