'''
Generate a complete PyTorch NN code that satisfies the following four Requirements: 

Architecture: CNN-1D 
Learning Task: Regression
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

class CNN1DTextRegressor(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings=120000, embedding_dim=128, padding_idx=0)
        self.conv1 = nn.Conv1d(in_channels=128, out_channels=192, kernel_size=7, stride=1, padding=3, bias=True)
        self.bn1 = nn.BatchNorm1d(num_features=192)
        self.relu1 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv1d(in_channels=192, out_channels=256, kernel_size=7, stride=1, padding=3, bias=True)
        self.bn2 = nn.BatchNorm1d(num_features=256)
        self.relu2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=5, stride=1, padding=2, bias=True)
        self.bn3 = nn.BatchNorm1d(num_features=256)
        self.relu3 = nn.ReLU(inplace=True)
        self.pool3 = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.global_pool = nn.AdaptiveAvgPool1d(output_size=1)
        self.dropout = nn.Dropout(p=0.2)
        self.fc1 = nn.Linear(in_features=256, out_features=64, bias=True)
        self.relu4 = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(in_features=64, out_features=1, bias=True)

    def forward(self, x):
        x = self.embedding(x)
        x = x.transpose(1, 2)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        x = self.pool3(x)
        x = self.global_pool(x)
        x = x.squeeze(-1)
        x = self.dropout(x)
        x = self.fc1(x)
        x = self.relu4(x)
        x = self.fc2(x)
        return x