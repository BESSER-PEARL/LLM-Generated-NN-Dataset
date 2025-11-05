'''
Generate a complete PyTorch NN code that satisfies the following four Requirements: 

Architecture: CNN-1D 
Learning Task: Classification-multiclass
Input Type + Scale: Text, seq_length: Large >2000 and vocab_size: Large >100k 
Complexity: Wide: embedding_dim of the Embedding layer at least min(128, upper_bound_of_vocab_size).


- Provide complete PyTorch code for the model only.
- All layer parameters must be fixed values inside the network, with no variables or external constants used.
- Avoid creating excessively large layers that would cause unreasonable memory usage.
- The model must only use standard layers directly, without defining helper functions or custom block classes.
- Output only the code, with no comments.
'''
import torch
import torch.nn as nn

class TextCNN1D(nn.Module):
    def __init__(self):
        super(TextCNN1D, self).__init__()
        self.embedding = nn.Embedding(150000, 128, padding_idx=0)
        self.embed_dropout = nn.Dropout(0.1)
        self.conv3 = nn.Conv1d(128, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(128)
        self.conv5 = nn.Conv1d(128, 128, kernel_size=5, padding=2)
        self.bn5 = nn.BatchNorm1d(128)
        self.conv9 = nn.Conv1d(128, 128, kernel_size=9, padding=4)
        self.bn9 = nn.BatchNorm1d(128)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.AdaptiveMaxPool1d(1)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(768, 256)
        self.dropout = nn.Dropout(0.2)
        self.fc_out = nn.Linear(256, 20)

    def forward(self, x):
        x = self.embedding(x)
        x = self.embed_dropout(x)
        x = x.transpose(1, 2)
        b1 = self.relu(self.bn3(self.conv3(x)))
        b2 = self.relu(self.bn5(self.conv5(x)))
        b3 = self.relu(self.bn9(self.conv9(x)))
        p1 = torch.cat([self.maxpool(b1), self.avgpool(b1)], dim=1)
        p2 = torch.cat([self.maxpool(b2), self.avgpool(b2)], dim=1)
        p3 = torch.cat([self.maxpool(b3), self.avgpool(b3)], dim=1)
        feats = torch.cat([p1, p2, p3], dim=1).squeeze(-1)
        out = self.fc1(feats)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc_out(out)
        return out