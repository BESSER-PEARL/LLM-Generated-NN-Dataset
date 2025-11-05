'''
Generate a complete PyTorch NN code that satisfies the following four Requirements: 

Architecture: CNN-1D 
Learning Task: Classification-multiclass
Input Type + Scale: Text, seq_length: Large >2000 and vocab_size: Large >100k 
Complexity: Wide-Deep: Number of CNN-1D layers at least 4, embedding_dim of the Embedding layer at least min(128, upper_bound_of_vocab_size).


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
        super().__init__()
        self.embedding = nn.Embedding(120000, 192, padding_idx=0)
        self.dropout_embed = nn.Dropout(0.1)
        self.conv1 = nn.Conv1d(192, 256, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm1d(256)
        self.conv2 = nn.Conv1d(256, 256, kernel_size=5, stride=2, padding=2)
        self.bn2 = nn.BatchNorm1d(256)
        self.conv3 = nn.Conv1d(256, 384, kernel_size=3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm1d(384)
        self.conv4 = nn.Conv1d(384, 384, kernel_size=3, stride=1, padding=2, dilation=2)
        self.bn4 = nn.BatchNorm1d(384)
        self.conv5 = nn.Conv1d(384, 512, kernel_size=3, stride=2, padding=1)
        self.bn5 = nn.BatchNorm1d(512)
        self.conv6 = nn.Conv1d(512, 512, kernel_size=3, stride=1, padding=4, dilation=4)
        self.bn6 = nn.BatchNorm1d(512)
        self.act = nn.GELU()
        self.dropout_conv = nn.Dropout(0.2)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(512, 256)
        self.bn_fc1 = nn.BatchNorm1d(256)
        self.dropout_fc = nn.Dropout(0.3)
        self.fc_out = nn.Linear(256, 20)

    def forward(self, x):
        x = self.embedding(x)
        x = self.dropout_embed(x)
        x = x.transpose(1, 2)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act(x)
        x = self.dropout_conv(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.act(x)
        x = self.dropout_conv(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.act(x)
        x = self.dropout_conv(x)
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.act(x)
        x = self.dropout_conv(x)
        x = self.conv5(x)
        x = self.bn5(x)
        x = self.act(x)
        x = self.dropout_conv(x)
        x = self.conv6(x)
        x = self.bn6(x)
        x = self.act(x)
        x = self.pool(x)
        x = x.squeeze(-1)
        x = self.fc1(x)
        x = self.bn_fc1(x)
        x = self.act(x)
        x = self.dropout_fc(x)
        x = self.fc_out(x)
        return x