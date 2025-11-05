'''
Generate a complete PyTorch NN code that satisfies the following four Requirements: 

Architecture: MLP 
Learning Task: Regression
Input Type + Scale: Tabular, features: Small <50 
Complexity: Wide: out_features of first linear layer at least min(128, upper_bound_of_features).


- Provide complete PyTorch code for the model only.
- All layer parameters must be fixed values inside the network, with no variables or external constants used.
- Avoid creating excessively large layers that would cause unreasonable memory usage.
- The model must only use standard layers directly, without defining helper functions or custom block classes.
- Output only the code, with no comments.
'''
import torch
import torch.nn as nn

class TabularMLPRegressor(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(32, 128),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(128),
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(128),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1)
        )
    def forward(self, x):
        return self.model(x)