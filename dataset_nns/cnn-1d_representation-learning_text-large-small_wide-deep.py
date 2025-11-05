'''
Generate a complete PyTorch NN code that satisfies the following four Requirements: 

Architecture: CNN-1D 
Learning Task: Representation-learning
Input Type + Scale: Text, seq_length: Large >2000 and vocab_size: Small <1k 
Complexity: Wide-Deep: Number of CNN-1D layers at least 4, embedding_dim of the Embedding layer at least min(128, upper_bound_of_vocab_size).


- Provide complete PyTorch code for the model only.
- All layer parameters must be fixed values inside the network, with no variables or external constants used.
- Avoid creating excessively large layers that would cause unreasonable memory usage.
- The model must only use standard layers directly, without defining helper functions or custom block classes.
- Output only the code, with no comments.
'''
import torch
import torch.nn as nn

class TextCNN1DWideDeep(nn.Module):
    def __init__(self):
        super(TextCNN1DWideDeep, self).__init__()
        self.embedding = nn.Embedding(num_embeddings=800, embedding_dim=192, padding_idx=0)
        self.dropout_embed = nn.Dropout(0.1)
        self.proj_in = nn.Conv1d(192, 256, kernel_size=1, stride=1, padding=0, dilation=1, bias=False)
        self.conv1 = nn.Conv1d(256, 256, kernel_size=5, stride=1, padding=2, dilation=1, bias=False)
        self.bn1 = nn.BatchNorm1d(256)
        self.conv2 = nn.Conv1d(256, 256, kernel_size=9, stride=1, padding=4, dilation=1, bias=False)
        self.bn2 = nn.BatchNorm1d(256)
        self.conv3 = nn.Conv1d(256, 256, kernel_size=3, stride=1, padding=2, dilation=2, bias=False)
        self.bn3 = nn.BatchNorm1d(256)
        self.conv4 = nn.Conv1d(256, 256, kernel_size=5, stride=1, padding=8, dilation=4, bias=False)
        self.bn4 = nn.BatchNorm1d(256)
        self.conv5 = nn.Conv1d(256, 256, kernel_size=9, stride=1, padding=4, dilation=1, bias=False)
        self.bn5 = nn.BatchNorm1d(256)
        self.conv6 = nn.Conv1d(256, 256, kernel_size=3, stride=1, padding=2, dilation=2, bias=False)
        self.bn6 = nn.BatchNorm1d(256)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(0.1)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.maxpool = nn.AdaptiveMaxPool1d(1)
        self.fc = nn.Linear(512, 256, bias=True)
        self.norm = nn.LayerNorm(256)

    def forward(self, x):
        x = self.embedding(x)
        x = self.dropout_embed(x)
        x = x.transpose(1, 2)
        res = self.proj_in(x)
        x = self.conv1(res)
        x = self.bn1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.act(x)
        x = self.dropout(x)
        x = x + res
        res = x
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.act(x)
        x = self.dropout(x)
        x = x + res
        res = x
        x = self.conv5(x)
        x = self.bn5(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.conv6(x)
        x = self.bn6(x)
        x = self.act(x)
        x = self.dropout(x)
        x = x + res
        avg = self.avgpool(x)
        mx = self.maxpool(x)
        x = torch.cat([avg, mx], dim=1)
        x = x.squeeze(-1)
        x = self.dropout(x)
        x = self.fc(x)
        x = self.norm(x)
        return x