'''
Generate a complete PyTorch NN code that satisfies the following four Requirements: 

Architecture: RNN-Simple 
Learning Task: Classification-multiclass
Input Type + Scale: Time series, seq_length: Large >2000 and features: Large >2000 
Complexity: Wide-Deep: Number of RNN-Simple layers at least 2, hidden_size of first RNN-Simple layer at least min(128, upper_bound_of_features).


- Provide complete PyTorch code for the model only.
- All layer parameters must be fixed values inside the network, with no variables or external constants used.
- Avoid creating excessively large layers that would cause unreasonable memory usage.
- The model must only use standard layers directly, without defining helper functions or custom block classes.
- Output only the code, with no comments.
'''
import torch
import torch.nn as nn

class RNNMulticlassClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.rnn1 = nn.RNN(input_size=2048, hidden_size=128, num_layers=1, batch_first=True, nonlinearity='tanh')
        self.rnn2 = nn.RNN(input_size=128, hidden_size=128, num_layers=1, batch_first=True, nonlinearity='tanh')
        self.fc_wide = nn.Linear(2048, 64)
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(p=0.2)
        self.fc_out = nn.Linear(128 + 64, 10)
    def forward(self, x):
        x_deep, _ = self.rnn1(x)
        x_deep, _ = self.rnn2(x_deep)
        deep_feat = x_deep[:, -1, :]
        wide_feat = x.mean(dim=1)
        wide_feat = self.act(self.fc_wide(wide_feat))
        feat = torch.cat([deep_feat, wide_feat], dim=1)
        feat = self.dropout(feat)
        logits = self.fc_out(feat)
        return logits