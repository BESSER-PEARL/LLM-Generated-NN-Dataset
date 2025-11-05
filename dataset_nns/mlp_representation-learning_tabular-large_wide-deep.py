'''
Generate a complete PyTorch NN code that satisfies the following four Requirements: 

Architecture: MLP 
Learning Task: Representation-learning
Input Type + Scale: Tabular, features: Large >2000 
Complexity: Wide-Deep: Number of linear layers at least 4, out_features of first linear layer at least min(128, upper_bound_of_features).


- Provide complete PyTorch code for the model only.
- All layer parameters must be fixed values inside the network, with no variables or external constants used.
- Avoid creating excessively large layers that would cause unreasonable memory usage.
- The model must only use standard layers directly, without defining helper functions or custom block classes.
- Output only the code, with no comments.
'''
import torch
import torch.nn as nn

class TabularRepresentationMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(2048, 512, bias=True)
        self.bn1 = nn.BatchNorm1d(512)
        self.drop1 = nn.Dropout(p=0.2)
        self.fc2 = nn.Linear(512, 1024, bias=True)
        self.bn2 = nn.BatchNorm1d(1024)
        self.drop2 = nn.Dropout(p=0.25)
        self.fc3 = nn.Linear(1024, 512, bias=True)
        self.bn3 = nn.BatchNorm1d(512)
        self.drop3 = nn.Dropout(p=0.2)
        self.fc4 = nn.Linear(512, 256, bias=True)
        self.bn4 = nn.BatchNorm1d(256)
        self.drop4 = nn.Dropout(p=0.15)
        self.fc5 = nn.Linear(256, 128, bias=True)
        self.bn5 = nn.BatchNorm1d(128)
        self.fc6 = nn.Linear(128, 64, bias=True)
        self.out_norm = nn.LayerNorm(64)
        self.act = nn.GELU()
    def forward(self, x):
        x = self.act(self.bn1(self.fc1(x)))
        x = self.drop1(x)
        x = self.act(self.bn2(self.fc2(x)))
        x = self.drop2(x)
        x = self.act(self.bn3(self.fc3(x)))
        x = self.drop3(x)
        x = self.act(self.bn4(self.fc4(x)))
        x = self.drop4(x)
        x = self.act(self.bn5(self.fc5(x)))
        x = self.fc6(x)
        x = self.out_norm(x)
        return x