'''
Generate a complete PyTorch NN code that satisfies the following four Requirements: 

Architecture: CNN-1D 
Learning Task: Classification-multiclass
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
import torch.nn.functional as F

class TextCNNWideDeep(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(120000, 128, padding_idx=0)
        self.dropout_embed = nn.Dropout(0.1)
        self.conv_d1 = nn.Conv1d(128, 128, kernel_size=3, padding=1, bias=False)
        self.bn_d1 = nn.BatchNorm1d(128)
        self.conv_d2 = nn.Conv1d(128, 128, kernel_size=3, padding=1, bias=False)
        self.bn_d2 = nn.BatchNorm1d(128)
        self.conv_d3 = nn.Conv1d(128, 128, kernel_size=3, padding=1, bias=False)
        self.bn_d3 = nn.BatchNorm1d(128)
        self.conv_d4 = nn.Conv1d(128, 128, kernel_size=3, padding=1, bias=False)
        self.bn_d4 = nn.BatchNorm1d(128)
        self.conv_w1 = nn.Conv1d(128, 64, kernel_size=1, padding=0)
        self.conv_w3 = nn.Conv1d(128, 64, kernel_size=3, padding=1)
        self.conv_w5 = nn.Conv1d(128, 64, kernel_size=5, padding=2)
        self.dropout_features = nn.Dropout(0.3)
        self.fc1 = nn.Linear(320, 256)
        self.bn_fc1 = nn.BatchNorm1d(256)
        self.fc_out = nn.Linear(256, 20)

    def forward(self, x):
        x = self.embedding(x)
        x = self.dropout_embed(x)
        x = x.transpose(1, 2)
        d = F.relu(self.bn_d1(self.conv_d1(x)))
        d = F.relu(self.bn_d2(self.conv_d2(d)))
        d = F.relu(self.bn_d3(self.conv_d3(d)))
        d = F.relu(self.bn_d4(self.conv_d4(d)))
        d = F.adaptive_max_pool1d(d, 1).squeeze(-1)
        w1 = F.relu(self.conv_w1(x))
        w3 = F.relu(self.conv_w3(x))
        w5 = F.relu(self.conv_w5(x))
        w1 = F.adaptive_max_pool1d(w1, 1).squeeze(-1)
        w3 = F.adaptive_max_pool1d(w3, 1).squeeze(-1)
        w5 = F.adaptive_max_pool1d(w5, 1).squeeze(-1)
        feat = torch.cat([d, w1, w3, w5], dim=1)
        feat = self.dropout_features(feat)
        feat = self.bn_fc1(self.fc1(feat))
        feat = F.relu(feat)
        logits = self.fc_out(feat)
        return logits