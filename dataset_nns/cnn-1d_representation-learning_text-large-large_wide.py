'''
Generate a complete PyTorch NN code that satisfies the following four Requirements: 

Architecture: CNN-1D 
Learning Task: Representation-learning
Input Type + Scale: Text, seq_length: Large >2000 and vocab_size: Large >100k 
Complexity: Wide: embedding_dim of the Embedding layer at least min(128, upper_bound_of_vocab_size).


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
        self.embedding = nn.Embedding(120000, 128, padding_idx=0)
        self.conv1 = nn.Conv1d(128, 256, kernel_size=5, stride=2, padding=2, bias=False)
        self.bn1 = nn.BatchNorm1d(256)
        self.conv2 = nn.Conv1d(256, 256, kernel_size=5, stride=2, padding=2, bias=False)
        self.bn2 = nn.BatchNorm1d(256)
        self.conv3 = nn.Conv1d(256, 512, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn3 = nn.BatchNorm1d(512)
        self.conv4 = nn.Conv1d(512, 512, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn4 = nn.BatchNorm1d(512)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.proj = nn.Linear(512, 256)
        self.act = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(0.1)
        self.norm = nn.LayerNorm(256)

    def forward(self, x):
        x = self.embedding(x)
        x = x.transpose(1, 2)
        x = self.bn1(self.conv1(x))
        x = self.act(x)
        x = self.dropout(x)
        x = self.bn2(self.conv2(x))
        x = self.act(x)
        x = self.dropout(x)
        x = self.bn3(self.conv3(x))
        x = self.act(x)
        x = self.dropout(x)
        x = self.bn4(self.conv4(x))
        x = self.act(x)
        x = self.pool(x).squeeze(-1)
        x = self.proj(x)
        x = self.norm(x)
        return x