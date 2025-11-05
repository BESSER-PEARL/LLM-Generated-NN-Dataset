'''
Generate a complete PyTorch NN code that satisfies the following four Requirements: 

Architecture: MLP 
Learning Task: Representation-learning
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

class TabularMLPRepresentation(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(40, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.act1 = nn.ReLU()
        self.drop1 = nn.Dropout(p=0.1)
        self.fc2 = nn.Linear(128, 160)
        self.bn2 = nn.BatchNorm1d(160)
        self.act2 = nn.ReLU()
        self.drop2 = nn.Dropout(p=0.1)
        self.fc3 = nn.Linear(160, 128)
        self.bn3 = nn.BatchNorm1d(128)
        self.act3 = nn.ReLU()
        self.drop3 = nn.Dropout(p=0.1)
        self.fc4 = nn.Linear(128, 96)
        self.bn4 = nn.BatchNorm1d(96)
        self.act4 = nn.ReLU()
        self.drop4 = nn.Dropout(p=0.1)
        self.fc5 = nn.Linear(96, 64)
        self.bn5 = nn.BatchNorm1d(64)
        self.act5 = nn.ReLU()
        self.drop5 = nn.Dropout(p=0.1)
        self.fc6 = nn.Linear(64, 32)

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
        x = self.drop3(x)
        x = self.fc4(x)
        x = self.bn4(x)
        x = self.act4(x)
        x = self.drop4(x)
        x = self.fc5(x)
        x = self.bn5(x)
        x = self.act5(x)
        x = self.drop5(x)
        x = self.fc6(x)
        return x