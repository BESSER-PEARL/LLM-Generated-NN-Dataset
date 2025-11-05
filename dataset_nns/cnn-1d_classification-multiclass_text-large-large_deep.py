'''
Generate a complete PyTorch NN code that satisfies the following four Requirements: 

Architecture: CNN-1D 
Learning Task: Classification-multiclass
Input Type + Scale: Text, seq_length: Large >2000 and vocab_size: Large >100k 
Complexity: Deep: Number of CNN-1D layers at least 4.


- Provide complete PyTorch code for the model only.
- All layer parameters must be fixed values inside the network, with no variables or external constants used.
- Avoid creating excessively large layers that would cause unreasonable memory usage.
- The model must only use standard layers directly, without defining helper functions or custom block classes.
- Output only the code, with no comments.
'''
import torch
import torch.nn as nn

class TextCNN1DDeep(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(120000, 128, padding_idx=0)
        self.conv1 = nn.Conv1d(128, 256, kernel_size=5, padding=2, stride=1, bias=False)
        self.bn1 = nn.BatchNorm1d(256)
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv1d(256, 256, kernel_size=5, padding=2, stride=1, bias=False)
        self.bn2 = nn.BatchNorm1d(256)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv1d(256, 384, kernel_size=3, padding=1, stride=1, bias=False)
        self.bn3 = nn.BatchNorm1d(384)
        self.pool3 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.conv4 = nn.Conv1d(384, 384, kernel_size=3, padding=1, stride=1, bias=False)
        self.bn4 = nn.BatchNorm1d(384)
        self.conv5 = nn.Conv1d(384, 512, kernel_size=3, padding=1, stride=1, bias=False)
        self.bn5 = nn.BatchNorm1d(512)
        self.pool5 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.conv6 = nn.Conv1d(512, 512, kernel_size=3, padding=1, stride=1, bias=False)
        self.bn6 = nn.BatchNorm1d(512)
        self.act = nn.ReLU(inplace=True)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.dropout1 = nn.Dropout(p=0.3)
        self.fc1 = nn.Linear(512, 256, bias=True)
        self.bn_fc = nn.BatchNorm1d(256)
        self.dropout2 = nn.Dropout(p=0.3)
        self.fc_out = nn.Linear(256, 12, bias=True)

    def forward(self, x):
        x = self.embedding(x)
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
        x = self.pool3(x)
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.act(x)
        x = self.conv5(x)
        x = self.bn5(x)
        x = self.act(x)
        x = self.pool5(x)
        x = self.conv6(x)
        x = self.bn6(x)
        x = self.act(x)
        x = self.global_pool(x)
        x = x.squeeze(-1)
        x = self.dropout1(x)
        x = self.fc1(x)
        x = self.bn_fc(x)
        x = self.act(x)
        x = self.dropout2(x)
        x = self.fc_out(x)
        return x