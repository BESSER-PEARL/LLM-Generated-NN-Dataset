'''
Generate a complete PyTorch NN code that satisfies the following four Requirements: 

Architecture: CNN-1D 
Learning Task: Regression
Input Type + Scale: Text, seq_length: Small <50 and vocab_size: Large >100k 
Complexity: Wide-Deep: Number of CNN-1D layers at least 4, embedding_dim of the Embedding layer at least min(128, upper_bound_of_vocab_size).


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
        self.embedding = nn.Embedding(num_embeddings=150000, embedding_dim=128, padding_idx=0)
        self.conv1 = nn.Conv1d(128, 192, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn1 = nn.BatchNorm1d(192)
        self.conv2 = nn.Conv1d(192, 192, kernel_size=5, stride=1, padding=2, bias=True)
        self.bn2 = nn.BatchNorm1d(192)
        self.conv3 = nn.Conv1d(192, 256, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn3 = nn.BatchNorm1d(256)
        self.conv4 = nn.Conv1d(256, 256, kernel_size=5, stride=1, padding=2, bias=True)
        self.bn4 = nn.BatchNorm1d(256)
        self.conv5 = nn.Conv1d(256, 320, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn5 = nn.BatchNorm1d(320)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(0.2)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.maxpool = nn.AdaptiveMaxPool1d(1)
        self.fc1 = nn.Linear(768, 128)
        self.fc_bn1 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        emb = self.embedding(x)
        wide_feat = emb.mean(dim=1)
        h = emb.transpose(1, 2)
        h = self.conv1(h)
        h = self.bn1(h)
        h = self.relu(h)
        h = self.dropout(h)
        h = self.conv2(h)
        h = self.bn2(h)
        h = self.relu(h)
        h = self.dropout(h)
        h = self.conv3(h)
        h = self.bn3(h)
        h = self.relu(h)
        h = self.dropout(h)
        h = self.conv4(h)
        h = self.bn4(h)
        h = self.relu(h)
        h = self.dropout(h)
        h = self.conv5(h)
        h = self.bn5(h)
        h = self.relu(h)
        h_avg = self.avgpool(h).squeeze(-1)
        h_max = self.maxpool(h).squeeze(-1)
        deep_feat = torch.cat([h_avg, h_max], dim=1)
        feat = torch.cat([deep_feat, wide_feat], dim=1)
        out = self.fc1(feat)
        out = self.fc_bn1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        return out