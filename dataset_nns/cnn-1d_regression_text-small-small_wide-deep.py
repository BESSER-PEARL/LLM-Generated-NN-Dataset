'''
Generate a complete PyTorch NN code that satisfies the following four Requirements: 

Architecture: CNN-1D 
Learning Task: Regression
Input Type + Scale: Text, seq_length: Small <50 and vocab_size: Small <1k 
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
        self.embedding = nn.Embedding(num_embeddings=800, embedding_dim=128, padding_idx=0)
        self.conv_d1 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=3, padding=1)
        self.bn_d1 = nn.BatchNorm1d(128)
        self.conv_d2 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=3, padding=1)
        self.bn_d2 = nn.BatchNorm1d(128)
        self.conv_d3 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=5, padding=2)
        self.bn_d3 = nn.BatchNorm1d(256)
        self.conv_d4 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3, padding=1)
        self.bn_d4 = nn.BatchNorm1d(256)
        self.conv_w1 = nn.Conv1d(in_channels=128, out_channels=64, kernel_size=1, padding=0)
        self.conv_w3 = nn.Conv1d(in_channels=128, out_channels=64, kernel_size=3, padding=1)
        self.conv_w5 = nn.Conv1d(in_channels=128, out_channels=64, kernel_size=5, padding=2)
        self.conv_w7 = nn.Conv1d(in_channels=128, out_channels=64, kernel_size=7, padding=3)
        self.dropout = nn.Dropout(p=0.2)
        self.fc1 = nn.Linear(in_features=512, out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=64)
        self.fc_out = nn.Linear(in_features=64, out_features=1)

    def forward(self, x):
        x = self.embedding(x)
        x = x.transpose(1, 2)
        d = F.relu(self.bn_d1(self.conv_d1(x)))
        d = F.relu(self.bn_d2(self.conv_d2(d)))
        d = F.relu(self.bn_d3(self.conv_d3(d)))
        d = F.relu(self.bn_d4(self.conv_d4(d)))
        d = torch.mean(d, dim=2)
        w1 = F.relu(self.conv_w1(x))
        w3 = F.relu(self.conv_w3(x))
        w5 = F.relu(self.conv_w5(x))
        w7 = F.relu(self.conv_w7(x))
        p1 = torch.amax(w1, dim=2)
        p3 = torch.amax(w3, dim=2)
        p5 = torch.amax(w5, dim=2)
        p7 = torch.amax(w7, dim=2)
        feats = torch.cat([d, p1, p3, p5, p7], dim=1)
        feats = F.relu(self.fc1(feats))
        feats = self.dropout(feats)
        feats = F.relu(self.fc2(feats))
        out = self.fc_out(feats)
        return out.squeeze(-1)