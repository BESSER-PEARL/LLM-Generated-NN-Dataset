'''
Generate a complete PyTorch NN code that satisfies the following four Requirements: 

Architecture: CNN-1D 
Learning Task: Representation-learning
Input Type + Scale: Text, seq_length: Small <50 and vocab_size: Large >100k 
Complexity: Wide: embedding_dim of the Embedding layer at least min(128, upper_bound_of_vocab_size).


- Provide complete PyTorch code for the model only.
- All layer parameters must be fixed values inside the network, with no variables or external constants used.
- Avoid creating excessively large layers that would cause unreasonable memory usage.
- The model must only use standard layers directly, without defining helper functions or custom block classes.
- Output only the code, with no comments.
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

class TextCNNRepresentation(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(120000, 128, padding_idx=0)
        self.dropout_embed = nn.Dropout(0.1)
        self.conv3 = nn.Conv1d(128, 160, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(160)
        self.conv5 = nn.Conv1d(128, 160, kernel_size=5, padding=2)
        self.bn5 = nn.BatchNorm1d(160)
        self.conv7 = nn.Conv1d(128, 160, kernel_size=7, padding=3)
        self.bn7 = nn.BatchNorm1d(160)
        self.act = nn.ReLU(inplace=True)
        self.dropout_conv = nn.Dropout(0.2)
        self.pool_max = nn.AdaptiveMaxPool1d(1)
        self.pool_avg = nn.AdaptiveAvgPool1d(1)
        self.proj1 = nn.Linear(960, 256)
        self.proj_bn = nn.BatchNorm1d(256)
        self.proj2 = nn.Linear(256, 256)
    def forward(self, x):
        x = self.embedding(x)
        x = self.dropout_embed(x)
        x = x.transpose(1, 2)
        c3 = self.conv3(x)
        c3 = self.bn3(c3)
        c3 = self.act(c3)
        c3 = self.dropout_conv(c3)
        c5 = self.conv5(x)
        c5 = self.bn5(c5)
        c5 = self.act(c5)
        c5 = self.dropout_conv(c5)
        c7 = self.conv7(x)
        c7 = self.bn7(c7)
        c7 = self.act(c7)
        c7 = self.dropout_conv(c7)
        c = torch.cat([c3, c5, c7], dim=1)
        m = self.pool_max(c).squeeze(-1)
        a = self.pool_avg(c).squeeze(-1)
        h = torch.cat([m, a], dim=1)
        h = self.proj1(h)
        h = self.proj_bn(h)
        h = self.act(h)
        h = self.proj2(h)
        h = F.normalize(h, p=2, dim=1)
        return h