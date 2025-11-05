'''
Generate a complete PyTorch NN code that satisfies the following four Requirements: 

Architecture: MLP 
Learning Task: Representation-learning
Input Type + Scale: Tabular, features: Large >2000 
Complexity: Deep: Number of linear layers at least 4.


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
        self.input_norm = nn.LayerNorm(4096, eps=1e-5, elementwise_affine=True)
        self.fc1 = nn.Linear(4096, 1024, bias=True)
        self.bn1 = nn.BatchNorm1d(1024, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True)
        self.act1 = nn.ReLU(inplace=True)
        self.drop1 = nn.Dropout(p=0.1)
        self.fc2 = nn.Linear(1024, 512, bias=True)
        self.bn2 = nn.BatchNorm1d(512, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True)
        self.act2 = nn.ReLU(inplace=True)
        self.drop2 = nn.Dropout(p=0.1)
        self.fc3 = nn.Linear(512, 256, bias=True)
        self.bn3 = nn.BatchNorm1d(256, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True)
        self.act3 = nn.ReLU(inplace=True)
        self.drop3 = nn.Dropout(p=0.1)
        self.fc4 = nn.Linear(256, 128, bias=True)
        self.bn4 = nn.BatchNorm1d(128, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True)
        self.act4 = nn.ReLU(inplace=True)
        self.fc5 = nn.Linear(128, 64, bias=True)
        self.out_norm = nn.LayerNorm(64, eps=1e-5, elementwise_affine=False)

    def forward(self, x):
        x = self.input_norm(x)
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
        x = self.fc5(x)
        x = self.out_norm(x)
        return x