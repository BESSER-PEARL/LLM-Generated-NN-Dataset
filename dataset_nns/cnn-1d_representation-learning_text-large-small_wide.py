'''
Generate a complete PyTorch NN code that satisfies the following four Requirements: 

Architecture: CNN-1D 
Learning Task: Representation-learning
Input Type + Scale: Text, seq_length: Large >2000 and vocab_size: Small <1k 
Complexity: Wide: embedding_dim of the Embedding layer at least min(128, upper_bound_of_vocab_size).


- Provide complete PyTorch code for the model only.
- All layer parameters must be fixed values inside the network, with no variables or external constants used.
- Avoid creating excessively large layers that would cause unreasonable memory usage.
- The model must only use standard layers directly, without defining helper functions or custom block classes.
- Output only the code, with no comments.
'''
import torch
import torch.nn as nn

class CNN1DTextEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings=999, embedding_dim=256, padding_idx=0)
        self.pre_ln = nn.LayerNorm(256)
        self.conv1 = nn.Conv1d(256, 256, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(256)
        self.conv2 = nn.Conv1d(256, 256, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn2 = nn.BatchNorm1d(256)
        self.conv3 = nn.Conv1d(256, 256, kernel_size=3, stride=1, dilation=2, padding=2, bias=False)
        self.bn3 = nn.BatchNorm1d(256)
        self.conv4 = nn.Conv1d(256, 256, kernel_size=3, stride=1, dilation=4, padding=4, bias=False)
        self.bn4 = nn.BatchNorm1d(256)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(p=0.1)
        self.proj = nn.Sequential(
            nn.Linear(256, 256, bias=True),
            nn.GELU(),
            nn.Dropout(p=0.1),
            nn.Linear(256, 192, bias=True),
            nn.LayerNorm(192)
        )
    def forward(self, x):
        x = self.embedding(x)
        x = self.pre_ln(x)
        x = x.transpose(1, 2)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act(x)
        x = self.dropout(x)
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
        x = x.mean(dim=2)
        x = self.proj(x)
        return x