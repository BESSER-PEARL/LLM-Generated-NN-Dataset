'''
Generate a complete PyTorch NN code that satisfies the following four Requirements: 

Architecture: RNN-GRU 
Learning Task: Classification-multiclass
Input Type + Scale: Time series, seq_length: Large >2000 and features: Large >2000 
Complexity: Wide-Deep: Number of RNN-GRU layers at least 2, hidden_size of first RNN-GRU layer at least min(128, upper_bound_of_features).


- Provide complete PyTorch code for the model only.
- All layer parameters must be fixed values inside the network, with no variables or external constants used.
- Avoid creating excessively large layers that would cause unreasonable memory usage.
- The model must only use standard layers directly, without defining helper functions or custom block classes.
- Output only the code, with no comments.
'''
import torch
import torch.nn as nn

class RNNGRUMulticlassClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.gru = nn.GRU(input_size=2048, hidden_size=256, num_layers=2, batch_first=True, dropout=0.2)
        self.fc1 = nn.Linear(256, 128)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.fc_out = nn.Linear(128, 10)
        self.wide_dropout = nn.Dropout(0.1)
        self.wide = nn.Linear(2048, 10)

    def forward(self, x):
        out, _ = self.gru(x)
        deep = out[:, -1, :]
        deep = self.fc1(deep)
        deep = self.relu(deep)
        deep = self.dropout(deep)
        deep_logits = self.fc_out(deep)
        wide_inp = x.mean(dim=1)
        wide_inp = self.wide_dropout(wide_inp)
        wide_logits = self.wide(wide_inp)
        return deep_logits + wide_logits