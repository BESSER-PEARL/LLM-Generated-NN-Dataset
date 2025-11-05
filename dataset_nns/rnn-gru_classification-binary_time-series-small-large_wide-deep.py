'''
Generate a complete PyTorch NN code that satisfies the following four Requirements: 

Architecture: RNN-GRU 
Learning Task: Classification-binary
Input Type + Scale: Time series, seq_length: Small <50 and features: Large >2000 
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
        self.gru1 = nn.GRU(input_size=2048, hidden_size=192, num_layers=1, batch_first=True)
        self.gru2 = nn.GRU(input_size=192, hidden_size=128, num_layers=1, batch_first=True)
        self.dropout_rnn = nn.Dropout(p=0.2)
        self.wide_fc1 = nn.Linear(2048, 256)
        self.wide_bn1 = nn.BatchNorm1d(256)
        self.dropout_wide = nn.Dropout(p=0.3)
        self.classifier = nn.Sequential(
            nn.Linear(384, 128),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        rnn_out1, _ = self.gru1(x)
        rnn_out1 = self.dropout_rnn(rnn_out1)
        rnn_out2, h2 = self.gru2(rnn_out1)
        deep_feat = h2[-1]
        wide_in = x.mean(dim=1)
        wide = self.wide_fc1(wide_in)
        wide = self.wide_bn1(wide)
        wide = torch.relu(wide)
        wide = self.dropout_wide(wide)
        combined = torch.cat([deep_feat, wide], dim=1)
        logits = self.classifier(combined)
        return logits