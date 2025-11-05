'''
Generate a complete PyTorch NN code that satisfies the following four Requirements: 

Architecture: CNN-1D 
Learning Task: Representation-learning
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

class TextCNNRepresentation(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings=100500, embedding_dim=128, padding_idx=0)
        self.conv1 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(256)
        self.conv2 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(256)
        self.conv3 = nn.Conv1d(in_channels=256, out_channels=384, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(384)
        self.conv4 = nn.Conv1d(in_channels=384, out_channels=512, kernel_size=5, padding=2)
        self.bn4 = nn.BatchNorm1d(512)
        self.conv5 = nn.Conv1d(in_channels=512, out_channels=256, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm1d(256)
        self.dropout = nn.Dropout(p=0.2)
        self.proj = nn.Linear(512, 256)
        self.out_norm = nn.LayerNorm(256)

    def forward(self, x):
        x = self.embedding(x)
        x = x.transpose(1, 2)
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
        avg = torch.mean(x, dim=2)
        mx = F.adaptive_max_pool1d(x, 1).squeeze(-1)
        h = torch.cat([avg, mx], dim=1)
        h = self.proj(h)
        h = self.out_norm(h)
        return h