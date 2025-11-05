'''
Generate a complete PyTorch NN code that satisfies the following four Requirements: 

Architecture: RNN-GRU 
Learning Task: Classification-multiclass
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

class GRUWideDeepClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.gru = nn.GRU(input_size=32, hidden_size=64, num_layers=2, batch_first=True)
        self.deep_fc = nn.Linear(64, 64)
        self.wide_proj = nn.Linear(32, 32)
        self.classifier = nn.Linear(96, 10)
        self.act = nn.ReLU()

    def forward(self, x):
        out, h = self.gru(x)
        deep = self.act(self.deep_fc(h[-1]))
        wide = self.act(self.wide_proj(x.mean(dim=1)))
        logits = self.classifier(torch.cat([deep, wide], dim=1))
        return logits