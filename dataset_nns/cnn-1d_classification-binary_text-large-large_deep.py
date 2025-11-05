'''
Generate a complete PyTorch NN code that satisfies the following four Requirements: 

Architecture: CNN-1D 
Learning Task: Classification-binary
Input Type + Scale: Text, seq_length: Large >2000 and vocab_size: Large >100k 
Complexity: Deep: Number of CNN-1D layers at least 4.


- Provide complete PyTorch code for the model only.
- All layer parameters must be fixed values inside the network, with no variables or external constants used.
- Avoid creating excessively large layers that would cause unreasonable memory usage.
- The model must only use standard layers directly, without defining helper functions or custom block classes.
- Output only the code, with no comments.
'''
import torch
import torch.nn as nn

class CNN1DBinaryTextClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(120000, 64, padding_idx=0)
        self.conv1 = nn.Conv1d(64, 128, kernel_size=5, stride=2, padding=2)
        self.bn1 = nn.BatchNorm1d(128)
        self.conv2 = nn.Conv1d(128, 128, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm1d(128)
        self.conv3 = nn.Conv1d(128, 192, kernel_size=3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm1d(192)
        self.conv4 = nn.Conv1d(192, 256, kernel_size=3, stride=2, padding=1)
        self.bn4 = nn.BatchNorm1d(256)
        self.conv5 = nn.Conv1d(256, 256, kernel_size=3, stride=1, padding=1)
        self.bn5 = nn.BatchNorm1d(256)
        self.conv6 = nn.Conv1d(256, 256, kernel_size=3, stride=1, padding=1)
        self.bn6 = nn.BatchNorm1d(256)
        self.relu = nn.ReLU(inplace=True)
        self.dropout1 = nn.Dropout(p=0.2)
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.fc1 = nn.Linear(256, 64)
        self.relu_fc = nn.ReLU(inplace=True)
        self.dropout2 = nn.Dropout(p=0.3)
        self.fc_out = nn.Linear(64, 1)

    def forward(self, x):
        x = self.embedding(x)
        x = x.transpose(1, 2)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu(x)
        x = self.conv5(x)
        x = self.bn5(x)
        x = self.relu(x)
        x = self.conv6(x)
        x = self.bn6(x)
        x = self.relu(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu_fc(x)
        x = self.dropout2(x)
        x = self.fc_out(x)
        return x