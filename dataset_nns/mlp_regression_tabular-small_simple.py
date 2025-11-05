'''
Generate a complete PyTorch NN code that satisfies the following four Requirements: 

Architecture: MLP 
Learning Task: Regression
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

class MLPRegressor(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(16, 32)
        self.act1 = nn.ReLU()
        self.fc2 = nn.Linear(32, 32)
        self.act2 = nn.ReLU()
        self.fc3 = nn.Linear(32, 16)
        self.act3 = nn.ReLU()
        self.out = nn.Linear(16, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act1(x)
        x = self.fc2(x)
        x = self.act2(x)
        x = self.fc3(x)
        x = self.act3(x)
        x = self.out(x)
        return x