'''
Generate a complete PyTorch NN code that satisfies the following four Requirements: 

Architecture: CNN-1D 
Learning Task: Regression
Input Type + Scale: Text, seq_length: Large >2000 and vocab_size: Small <1k 
Complexity: Wide: embedding_dim of the Embedding layer at least min(128, upper_bound_of_vocab_size).


- Provide complete PyTorch code for the model only.
- All layer parameters must be fixed values inside the network, with no variables or external constants used.
- Avoid creating excessively large layers that would cause unreasonable memory usage.
- The model must only use standard layers directly, without defining helper functions or custom block classes.
- Output only the code, with no comments.
'''
import torch
import torch.nn as nn

class CNN1DRegressor(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings=800, embedding_dim=128, padding_idx=0)
        self.embed_dropout = nn.Dropout(p=0.2)
        self.conv3 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=3, padding=1, stride=1, bias=True)
        self.bn3 = nn.BatchNorm1d(num_features=128)
        self.conv5 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=5, padding=2, stride=1, bias=True)
        self.bn5 = nn.BatchNorm1d(num_features=128)
        self.conv7 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=7, padding=3, stride=1, bias=True)
        self.bn7 = nn.BatchNorm1d(num_features=128)
        self.act = nn.ReLU(inplace=True)
        self.pool = nn.AdaptiveMaxPool1d(output_size=1)
        self.fc1 = nn.Linear(in_features=384, out_features=128, bias=True)
        self.bn_fc = nn.BatchNorm1d(num_features=128)
        self.fc_dropout = nn.Dropout(p=0.2)
        self.fc_out = nn.Linear(in_features=128, out_features=1, bias=True)

    def forward(self, x):
        x = self.embedding(x)
        x = self.embed_dropout(x)
        x = x.transpose(1, 2)
        c3 = self.conv3(x)
        c3 = self.bn3(c3)
        c3 = self.act(c3)
        c3 = self.pool(c3).squeeze(-1)
        c5 = self.conv5(x)
        c5 = self.bn5(c5)
        c5 = self.act(c5)
        c5 = self.pool(c5).squeeze(-1)
        c7 = self.conv7(x)
        c7 = self.bn7(c7)
        c7 = self.act(c7)
        c7 = self.pool(c7).squeeze(-1)
        h = torch.cat([c3, c5, c7], dim=1)
        h = self.fc1(h)
        h = self.bn_fc(h)
        h = self.act(h)
        h = self.fc_dropout(h)
        y = self.fc_out(h)
        return y