'''
Generate a complete PyTorch NN code that satisfies the following four Requirements: 

Architecture: RNN-GRU 
Learning Task: Classification-binary
Input Type + Scale: Time series, seq_length: Large >2000 and features: Small <50 
Complexity: Wide-Deep: Number of RNN-GRU layers at least 2, hidden_size of first RNN-GRU layer at least min(128, upper_bound_of_features).


- Provide complete PyTorch code for the model only.
- All layer parameters must be fixed values inside the network, with no variables or external constants used.
- Avoid creating excessively large layers that would cause unreasonable memory usage.
- The model must only use standard layers directly, without defining helper functions or custom block classes.
- Output only the code, with no comments.
'''
import torch
import torch.nn as nn

class GRUWideDeepBinaryClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.gru = nn.GRU(input_size=32, hidden_size=64, num_layers=2, batch_first=True, dropout=0.1, bidirectional=False)
        self.deep_fc1 = nn.Linear(64, 32)
        self.deep_act1 = nn.ReLU()
        self.deep_drop1 = nn.Dropout(p=0.2)
        self.wide_fc = nn.Linear(64, 32)
        self.wide_act = nn.ReLU()
        self.classifier = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        _, h_n = self.gru(x)
        deep = h_n[-1]
        deep = self.deep_fc1(deep)
        deep = self.deep_act1(deep)
        deep = self.deep_drop1(deep)
        wide_mean = x.mean(dim=1)
        wide_max = x.max(dim=1).values
        wide = torch.cat([wide_mean, wide_max], dim=1)
        wide = self.wide_fc(wide)
        wide = self.wide_act(wide)
        fused = torch.cat([deep, wide], dim=1)
        out = self.classifier(fused)
        return out