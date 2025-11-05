'''
Generate a complete PyTorch NN code that satisfies the following four Requirements: 

Architecture: CNN-1D 
Learning Task: Classification-binary
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

class TextCNN1DBinary(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(120000, 128, padding_idx=0)
        self.dropout_embed = nn.Dropout(0.15)
        self.conv1 = nn.Conv1d(128, 160, 5, padding=2)
        self.bn1 = nn.BatchNorm1d(160)
        self.pool1 = nn.MaxPool1d(4, stride=4)
        self.conv2 = nn.Conv1d(160, 224, 5, padding=2)
        self.bn2 = nn.BatchNorm1d(224)
        self.pool2 = nn.MaxPool1d(4, stride=4)
        self.conv3 = nn.Conv1d(224, 256, 3, padding=1)
        self.bn3 = nn.BatchNorm1d(256)
        self.act = nn.ReLU(inplace=True)
        self.global_pool = nn.AdaptiveMaxPool1d(1)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(256, 1)

    def forward(self, x):
        x = self.embedding(x)
        x = self.dropout_embed(x)
        x = x.transpose(1, 2)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.act(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.act(x)
        x = self.global_pool(x)
        x = x.squeeze(-1)
        x = self.dropout(x)
        x = self.fc(x)
        return x