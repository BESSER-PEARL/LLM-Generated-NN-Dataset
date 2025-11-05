'''
Generate a complete PyTorch NN code that satisfies the following four Requirements: 

Architecture: CNN-1D 
Learning Task: Classification-multiclass
Input Type + Scale: Time series, seq_length: Large >2000 and features: Large >2000 
Complexity: Wide: out_channels of first Conv layer at least min(128, upper_bound_of_features).


- Provide complete PyTorch code for the model only.
- All layer parameters must be fixed values inside the network, with no variables or external constants used.
- Avoid creating excessively large layers that would cause unreasonable memory usage.
- The model must only use standard layers directly, without defining helper functions or custom block classes.
- Output only the code, with no comments.
'''
import torch
import torch.nn as nn

class TimeSeriesCNN1D(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(2048, 256, kernel_size=7, stride=4, padding=3, bias=False),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Conv1d(256, 384, kernel_size=5, stride=2, padding=2, bias=False),
            nn.BatchNorm1d(384),
            nn.GELU(),
            nn.Conv1d(384, 512, kernel_size=5, stride=2, padding=2, bias=False),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Conv1d(512, 512, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=0.2),
            nn.Linear(512, 10)
        )
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x