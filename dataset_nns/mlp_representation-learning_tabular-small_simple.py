'''
Generate a complete PyTorch NN code that satisfies the following four Requirements: 

Architecture: MLP 
Learning Task: Representation-learning
Input Type + Scale: Tabular, features: Small <50 
Complexity: Simple: Number of linear layers up to 4, out_features of first linear layer up to min(128, upper_bound_of_features).


- Provide complete PyTorch code for the model only.
- All layer parameters must be fixed values inside the network, with no variables or external constants used.
- Avoid creating excessively large layers that would cause unreasonable memory usage.
- The model must only use standard layers directly, without defining helper functions or custom block classes.
- Output only the code, with no comments.
'''
import torch
import torch.nn as nn

class TabularMLPEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(32, 32, bias=True)
        self.bn1 = nn.BatchNorm1d(32)
        self.act1 = nn.ReLU()
        self.drop1 = nn.Dropout(0.1)
        self.fc2 = nn.Linear(32, 24, bias=True)
        self.bn2 = nn.BatchNorm1d(24)
        self.act2 = nn.ReLU()
        self.drop2 = nn.Dropout(0.1)
        self.fc3 = nn.Linear(24, 16, bias=True)
        self.bn3 = nn.BatchNorm1d(16)
        self.act3 = nn.ReLU()
        self.fc4 = nn.Linear(16, 16, bias=True)

    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.act2(x)
        x = self.drop2(x)
        x = self.fc3(x)
        x = self.bn3(x)
        x = self.act3(x)
        x = self.fc4(x)
        return x