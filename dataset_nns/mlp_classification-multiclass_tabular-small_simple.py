'''
Generate a complete PyTorch NN code that satisfies the following four Requirements: 

Architecture: MLP 
Learning Task: Classification-multiclass
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

class TabularMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Linear(20, 16, bias=True),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(16),
            nn.Dropout(p=0.1),
            nn.Linear(16, 16, bias=True),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(16),
            nn.Dropout(p=0.1),
            nn.Linear(16, 8, bias=True),
            nn.ReLU(inplace=True)
        )
        self.classifier = nn.Linear(8, 10, bias=True)

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x