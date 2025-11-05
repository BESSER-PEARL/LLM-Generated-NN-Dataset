'''
Generate a complete PyTorch NN code that satisfies the following four Requirements: 

Architecture: CNN-1D 
Learning Task: Representation-learning
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
import torch.nn.functional as F

class TextCNN1DRepresentation(nn.Module):
    def __init__(self):
        super(TextCNN1DRepresentation, self).__init__()
        self.embedding = nn.Embedding(131072, 128, padding_idx=0)
        self.conv1 = nn.Conv1d(128, 256, kernel_size=5, stride=2, padding=2, bias=False)
        self.bn1 = nn.BatchNorm1d(256)
        self.relu1 = nn.ReLU(inplace=True)
        self.drop1 = nn.Dropout(p=0.1)
        self.conv2 = nn.Conv1d(256, 256, kernel_size=5, stride=2, padding=2, bias=False)
        self.bn2 = nn.BatchNorm1d(256)
        self.relu2 = nn.ReLU(inplace=True)
        self.drop2 = nn.Dropout(p=0.1)
        self.conv3 = nn.Conv1d(256, 384, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn3 = nn.BatchNorm1d(384)
        self.relu3 = nn.ReLU(inplace=True)
        self.drop3 = nn.Dropout(p=0.1)
        self.conv4 = nn.Conv1d(384, 512, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn4 = nn.BatchNorm1d(512)
        self.relu4 = nn.ReLU(inplace=True)
        self.drop4 = nn.Dropout(p=0.1)
        self.conv5 = nn.Conv1d(512, 512, kernel_size=3, stride=1, padding=2, dilation=2, bias=False)
        self.bn5 = nn.BatchNorm1d(512)
        self.relu5 = nn.ReLU(inplace=True)
        self.drop5 = nn.Dropout(p=0.1)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.maxpool = nn.AdaptiveMaxPool1d(1)
        self.fc = nn.Linear(1024, 256)
        self.out_bn = nn.BatchNorm1d(256)
        self.out_drop = nn.Dropout(p=0.1)

    def forward(self, x):
        x = self.embedding(x)
        x = x.transpose(1, 2)
        x = self.drop1(self.relu1(self.bn1(self.conv1(x))))
        x = self.drop2(self.relu2(self.bn2(self.conv2(x))))
        x = self.drop3(self.relu3(self.bn3(self.conv3(x))))
        x = self.drop4(self.relu4(self.bn4(self.conv4(x))))
        x = self.drop5(self.relu5(self.bn5(self.conv5(x))))
        avg = self.avgpool(x).squeeze(-1)
        mx = self.maxpool(x).squeeze(-1)
        h = torch.cat([avg, mx], dim=1)
        h = self.fc(h)
        h = self.out_bn(h)
        h = F.relu(h, inplace=True)
        h = self.out_drop(h)
        return h