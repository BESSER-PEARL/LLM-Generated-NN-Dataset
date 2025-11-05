'''
Generate a complete PyTorch NN code that satisfies the following four Requirements: 

Architecture: MLP 
Learning Task: Classification-binary
Input Type + Scale: Tabular, features: Small <50 
Complexity: Wide-Deep: Number of linear layers at least 4, out_features of first linear layer at least min(128, upper_bound_of_features).


- Provide complete PyTorch code for the model only.
- All layer parameters must be fixed values inside the network, with no variables or external constants used.
- Avoid creating excessively large layers that would cause unreasonable memory usage.
- The model must only use standard layers directly, without defining helper functions or custom block classes.
- Output only the code, with no comments.
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

class TabularBinaryMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(32, 128, bias=True)
        self.bn1 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 128, bias=True)
        self.bn2 = nn.BatchNorm1d(128)
        self.fc3 = nn.Linear(128, 64, bias=True)
        self.bn3 = nn.BatchNorm1d(64)
        self.fc4 = nn.Linear(64, 64, bias=True)
        self.bn4 = nn.BatchNorm1d(64)
        self.fc5 = nn.Linear(64, 32, bias=True)
        self.bn5 = nn.BatchNorm1d(32)
        self.dropout = nn.Dropout(p=0.2)
        self.out = nn.Linear(32, 1, bias=True)

    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = F.relu(x, inplace=True)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = F.relu(x, inplace=True)
        x = self.dropout(x)
        x = self.fc3(x)
        x = self.bn3(x)
        x = F.relu(x, inplace=True)
        x = self.dropout(x)
        x = self.fc4(x)
        x = self.bn4(x)
        x = F.relu(x, inplace=True)
        x = self.dropout(x)
        x = self.fc5(x)
        x = self.bn5(x)
        x = F.relu(x, inplace=True)
        x = self.out(x)
        return x