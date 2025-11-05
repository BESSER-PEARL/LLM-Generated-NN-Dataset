'''
Generate a complete PyTorch NN code that satisfies the following four Requirements: 

Architecture: CNN-1D 
Learning Task: Classification-binary
Input Type + Scale: Text, seq_length: Large >2000 and vocab_size: Large >100k 
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

class WideDeepTextCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings=150000, embedding_dim=128, padding_idx=0)
        self.embed_dropout = nn.Dropout(p=0.2)
        self.conv1 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=5, padding=2, stride=1, dilation=1, bias=False)
        self.bn1 = nn.BatchNorm1d(128)
        self.conv2 = nn.Conv1d(in_channels=128, out_channels=192, kernel_size=5, padding=4, stride=1, dilation=2, bias=False)
        self.bn2 = nn.BatchNorm1d(192)
        self.conv3 = nn.Conv1d(in_channels=192, out_channels=256, kernel_size=3, padding=2, stride=1, dilation=2, bias=False)
        self.bn3 = nn.BatchNorm1d(256)
        self.conv4 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3, padding=4, stride=1, dilation=4, bias=False)
        self.bn4 = nn.BatchNorm1d(256)
        self.conv_w3 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=3, padding=1, bias=False)
        self.bn_w3 = nn.BatchNorm1d(128)
        self.conv_w5 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=5, padding=2, bias=False)
        self.bn_w5 = nn.BatchNorm1d(128)
        self.conv_w7 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=7, padding=3, bias=False)
        self.bn_w7 = nn.BatchNorm1d(128)
        self.dropout = nn.Dropout(p=0.5)
        self.fc1 = nn.Linear(256 + 128 + 128 + 128, 256)
        self.fc1_bn = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 1)

    def forward(self, x):
        x = self.embedding(x)
        x = self.embed_dropout(x)
        x = x.transpose(1, 2)
        d = self.conv1(x)
        d = self.bn1(d)
        d = F.relu(d)
        d = self.conv2(d)
        d = self.bn2(d)
        d = F.relu(d)
        d = self.conv3(d)
        d = self.bn3(d)
        d = F.relu(d)
        d = self.conv4(d)
        d = self.bn4(d)
        d = F.relu(d)
        d = F.adaptive_max_pool1d(d, 1).squeeze(-1)
        w3 = self.conv_w3(x)
        w3 = self.bn_w3(w3)
        w3 = F.relu(w3)
        w3 = F.adaptive_max_pool1d(w3, 1).squeeze(-1)
        w5 = self.conv_w5(x)
        w5 = self.bn_w5(w5)
        w5 = F.relu(w5)
        w5 = F.adaptive_max_pool1d(w5, 1).squeeze(-1)
        w7 = self.conv_w7(x)
        w7 = self.bn_w7(w7)
        w7 = F.relu(w7)
        w7 = F.adaptive_max_pool1d(w7, 1).squeeze(-1)
        h = torch.cat([d, w3, w5, w7], dim=1)
        h = self.dropout(h)
        h = self.fc1(h)
        h = self.fc1_bn(h)
        h = F.relu(h)
        h = self.dropout(h)
        out = self.fc2(h)
        return out