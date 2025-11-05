'''
Generate a complete PyTorch NN code that satisfies the following four Requirements: 

Architecture: CNN-1D 
Learning Task: Representation-learning
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

class TextCNNRepresentation(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings=800, embedding_dim=128, padding_idx=0)
        self.conv1 = nn.Conv1d(in_channels=128, out_channels=160, kernel_size=3, padding=1, stride=1, bias=True)
        self.bn1 = nn.BatchNorm1d(num_features=160)
        self.conv2 = nn.Conv1d(in_channels=160, out_channels=192, kernel_size=3, padding=1, stride=1, bias=True)
        self.bn2 = nn.BatchNorm1d(num_features=192)
        self.conv3 = nn.Conv1d(in_channels=192, out_channels=224, kernel_size=3, padding=1, stride=1, bias=True)
        self.bn3 = nn.BatchNorm1d(num_features=224)
        self.conv4 = nn.Conv1d(in_channels=224, out_channels=256, kernel_size=3, padding=1, stride=1, bias=True)
        self.bn4 = nn.BatchNorm1d(num_features=256)
        self.conv5 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3, padding=1, stride=1, bias=True)
        self.bn5 = nn.BatchNorm1d(num_features=256)
        self.conv6 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3, padding=1, stride=1, bias=True)
        self.bn6 = nn.BatchNorm1d(num_features=256)
        self.act = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=0.1)
        self.pool = nn.AdaptiveAvgPool1d(output_size=1)
        self.proj = nn.Linear(in_features=256, out_features=256, bias=True)

    def forward(self, x):
        x = self.embedding(x)
        x = x.transpose(1, 2)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.act(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.act(x)
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.act(x)
        x = self.conv5(x)
        x = self.bn5(x)
        x = self.act(x)
        x = self.conv6(x)
        x = self.bn6(x)
        x = self.act(x)
        x = self.pool(x).squeeze(-1)
        x = self.dropout(x)
        x = self.proj(x)
        return x