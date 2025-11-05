'''
Generate a complete PyTorch NN code that satisfies the following four Requirements: 

Architecture: CNN-1D 
Learning Task: Regression
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

class TextCNNRegressor(nn.Module):
    def __init__(self):
        super(TextCNNRegressor, self).__init__()
        self.embedding = nn.Embedding(200000, 128, padding_idx=0)
        self.conv1 = nn.Conv1d(128, 192, kernel_size=5, stride=1, padding=2)
        self.bn1 = nn.BatchNorm1d(192)
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv1d(192, 192, kernel_size=5, stride=1, padding=2)
        self.bn2 = nn.BatchNorm1d(192)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv1d(192, 256, kernel_size=3, stride=1, padding=2, dilation=2)
        self.bn3 = nn.BatchNorm1d(256)
        self.conv4 = nn.Conv1d(256, 256, kernel_size=3, stride=1, padding=3, dilation=3)
        self.bn4 = nn.BatchNorm1d(256)
        self.pool3 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.conv5 = nn.Conv1d(256, 192, kernel_size=1, stride=1, padding=0)
        self.bn5 = nn.BatchNorm1d(192)
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(0.2)
        self.fc1 = nn.Linear(192, 128)
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        x = self.embedding(x)
        x = x.transpose(1, 2)
        x = self.conv1(x)
        x = self.bn1(x)
        x = torch.relu(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = torch.relu(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = torch.relu(x)
        x = self.conv4(x)
        x = self.bn4(x)
        x = torch.relu(x)
        x = self.pool3(x)
        x = self.conv5(x)
        x = self.bn5(x)
        x = torch.relu(x)
        x = self.gap(x).squeeze(-1)
        x = self.dropout(x)
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x.squeeze(-1)