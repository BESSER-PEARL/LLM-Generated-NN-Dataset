'''
Generate a complete PyTorch NN code that satisfies the following four Requirements: 

Architecture: RNN-LSTM 
Learning Task: Representation-learning
Input Type + Scale: Time series, seq_length: Large >2000 and features: Large >2000 
Complexity: Wide-Deep: Number of RNN-LSTM layers at least 2, hidden_size of first RNN-LSTM layer at least min(128, upper_bound_of_features).


- Provide complete PyTorch code for the model only.
- All layer parameters must be fixed values inside the network, with no variables or external constants used.
- Avoid creating excessively large layers that would cause unreasonable memory usage.
- The model must only use standard layers directly, without defining helper functions or custom block classes.
- Output only the code, with no comments.
'''
import torch
import torch.nn as nn

class WideDeepLSTMEncoder(nn.Module):
    def __init__(self):
        super(WideDeepLSTMEncoder, self).__init__()
        self.lstm1 = nn.LSTM(input_size=2048, hidden_size=256, num_layers=1, batch_first=True, bidirectional=False)
        self.lstm2 = nn.LSTM(input_size=256, hidden_size=128, num_layers=1, batch_first=True, bidirectional=False)
        self.deep_norm = nn.LayerNorm(128)
        self.deep_linear = nn.Linear(128, 64)
        self.wide_linear = nn.Linear(2048, 64)
        self.fuse_norm = nn.LayerNorm(128)
        self.fuse_linear = nn.Linear(128, 128)
        self.output_norm = nn.LayerNorm(128)
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, x):
        wide_feat = x.mean(dim=1)
        wide_feat = self.wide_linear(wide_feat)
        wide_feat = self.act(wide_feat)
        wide_feat = self.dropout(wide_feat)
        out1, _ = self.lstm1(x)
        out1 = self.dropout(out1)
        out2, _ = self.lstm2(out1)
        deep_feat = out2.mean(dim=1)
        deep_feat = self.deep_norm(deep_feat)
        deep_feat = self.deep_linear(deep_feat)
        deep_feat = self.act(deep_feat)
        deep_feat = self.dropout(deep_feat)
        fused = torch.cat([deep_feat, wide_feat], dim=-1)
        fused = self.fuse_norm(fused)
        emb = self.fuse_linear(fused)
        emb = self.act(emb)
        emb = self.output_norm(emb)
        return emb