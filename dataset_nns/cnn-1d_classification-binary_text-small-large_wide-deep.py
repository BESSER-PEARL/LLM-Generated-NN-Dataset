'''
Generate a complete PyTorch NN code that satisfies the following four Requirements: 

Architecture: CNN-1D 
Learning Task: Classification-binary
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

class TextCNNBinary(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings=120000, embedding_dim=128, padding_idx=0)
        self.dropout_embed = nn.Dropout(p=0.2)
        self.conv1a = nn.Conv1d(in_channels=128, out_channels=32, kernel_size=3, padding=1, stride=1, dilation=1, groups=1, bias=True)
        self.bn1a = nn.BatchNorm1d(num_features=32)
        self.conv1b = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=3, padding=1, stride=1, dilation=1, groups=1, bias=True)
        self.bn1b = nn.BatchNorm1d(num_features=32)
        self.conv2a = nn.Conv1d(in_channels=128, out_channels=32, kernel_size=5, padding=2, stride=1, dilation=1, groups=1, bias=True)
        self.bn2a = nn.BatchNorm1d(num_features=32)
        self.conv2b = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=5, padding=2, stride=1, dilation=1, groups=1, bias=True)
        self.bn2b = nn.BatchNorm1d(num_features=32)
        self.conv3a = nn.Conv1d(in_channels=128, out_channels=32, kernel_size=7, padding=3, stride=1, dilation=1, groups=1, bias=True)
        self.bn3a = nn.BatchNorm1d(num_features=32)
        self.conv3b = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=7, padding=3, stride=1, dilation=1, groups=1, bias=True)
        self.bn3b = nn.BatchNorm1d(num_features=32)
        self.conv4a = nn.Conv1d(in_channels=128, out_channels=32, kernel_size=9, padding=4, stride=1, dilation=1, groups=1, bias=True)
        self.bn4a = nn.BatchNorm1d(num_features=32)
        self.conv4b = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=9, padding=4, stride=1, dilation=1, groups=1, bias=True)
        self.bn4b = nn.BatchNorm1d(num_features=32)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.AdaptiveMaxPool1d(output_size=1)
        self.fc1 = nn.Linear(in_features=128, out_features=64, bias=True)
        self.bn_fc1 = nn.BatchNorm1d(num_features=64)
        self.dropout_fc = nn.Dropout(p=0.3)
        self.fc2 = nn.Linear(in_features=64, out_features=1, bias=True)

    def forward(self, x):
        x = self.embedding(x)
        x = self.dropout_embed(x)
        x = x.transpose(1, 2)
        y1 = self.conv1a(x)
        y1 = self.bn1a(y1)
        y1 = self.relu(y1)
        y1 = self.conv1b(y1)
        y1 = self.bn1b(y1)
        y1 = self.relu(y1)
        y1 = self.pool(y1).squeeze(-1)
        y2 = self.conv2a(x)
        y2 = self.bn2a(y2)
        y2 = self.relu(y2)
        y2 = self.conv2b(y2)
        y2 = self.bn2b(y2)
        y2 = self.relu(y2)
        y2 = self.pool(y2).squeeze(-1)
        y3 = self.conv3a(x)
        y3 = self.bn3a(y3)
        y3 = self.relu(y3)
        y3 = self.conv3b(y3)
        y3 = self.bn3b(y3)
        y3 = self.relu(y3)
        y3 = self.pool(y3).squeeze(-1)
        y4 = self.conv4a(x)
        y4 = self.bn4a(y4)
        y4 = self.relu(y4)
        y4 = self.conv4b(y4)
        y4 = self.bn4b(y4)
        y4 = self.relu(y4)
        y4 = self.pool(y4).squeeze(-1)
        y = torch.cat([y1, y2, y3, y4], dim=1)
        y = self.fc1(y)
        y = self.bn_fc1(y)
        y = self.relu(y)
        y = self.dropout_fc(y)
        y = self.fc2(y)
        return y.squeeze(-1)