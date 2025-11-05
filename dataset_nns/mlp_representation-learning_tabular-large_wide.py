'''
Generate a complete PyTorch NN code that satisfies the following four Requirements: 

Architecture: MLP 
Learning Task: Representation-learning
Input Type + Scale: Tabular, features: Large >2000 
Complexity: Wide: out_features of first linear layer at least min(128, upper_bound_of_features).


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
        self.input_bn = nn.BatchNorm1d(2048, affine=False)
        self.fc1 = nn.Linear(2048, 512, bias=True)
        self.bn1 = nn.BatchNorm1d(512)
        self.dropout1 = nn.Dropout(0.1)
        self.fc2 = nn.Linear(512, 256, bias=True)
        self.bn2 = nn.BatchNorm1d(256)
        self.dropout2 = nn.Dropout(0.1)
        self.fc3 = nn.Linear(256, 128, bias=True)
        self.out_norm = nn.LayerNorm(128)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.input_bn(x)
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.act(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.act(x)
        x = self.dropout2(x)
        x = self.fc3(x)
        x = self.out_norm(x)
        return x