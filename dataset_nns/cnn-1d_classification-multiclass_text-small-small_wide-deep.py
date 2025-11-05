'''
Generate a complete PyTorch NN code that satisfies the following four Requirements: 

Architecture: CNN-1D 
Learning Task: Classification-multiclass
Input Type + Scale: Text, seq_length: Small <50 and vocab_size: Small <1k 
Complexity: Wide-Deep: Number of CNN-1D layers at least 4, embedding_dim of the Embedding layer at least min(128, upper_bound_of_vocab_size).


- Provide complete PyTorch code for the model only.
- All layer parameters must be fixed values inside the network, with no variables or external constants used.
- Avoid creating excessively large layers that would cause unreasonable memory usage.
- The model must only use standard layers directly, without defining helper functions or custom block classes.
- Output only the code, with no comments.
'''
import torch
import torch.nn as nn

class WideDeepTextCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings=800, embedding_dim=128, padding_idx=0)
        self.dropout_embed = nn.Dropout(p=0.2)
        self.branch3_conv1 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=3, padding=1)
        self.branch3_bn1 = nn.BatchNorm1d(num_features=128)
        self.branch3_relu1 = nn.ReLU()
        self.branch3_conv2 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=3, padding=1)
        self.branch3_bn2 = nn.BatchNorm1d(num_features=128)
        self.branch3_relu2 = nn.ReLU()
        self.branch5_conv1 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=5, padding=2)
        self.branch5_bn1 = nn.BatchNorm1d(num_features=128)
        self.branch5_relu1 = nn.ReLU()
        self.branch5_conv2 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=5, padding=2)
        self.branch5_bn2 = nn.BatchNorm1d(num_features=128)
        self.branch5_relu2 = nn.ReLU()
        self.branch7_conv1 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=7, padding=3)
        self.branch7_bn1 = nn.BatchNorm1d(num_features=128)
        self.branch7_relu1 = nn.ReLU()
        self.branch7_conv2 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=7, padding=3)
        self.branch7_bn2 = nn.BatchNorm1d(num_features=128)
        self.branch7_relu2 = nn.ReLU()
        self.pool = nn.AdaptiveMaxPool1d(output_size=1)
        self.fc1 = nn.Linear(in_features=384, out_features=192)
        self.fc1_relu = nn.ReLU()
        self.fc1_dropout = nn.Dropout(p=0.3)
        self.fc2 = nn.Linear(in_features=192, out_features=8)
    def forward(self, x):
        x = self.embedding(x)
        x = self.dropout_embed(x)
        x = x.transpose(1, 2)
        a = self.branch3_conv1(x)
        a = self.branch3_bn1(a)
        a = self.branch3_relu1(a)
        a = self.branch3_conv2(a)
        a = self.branch3_bn2(a)
        a = self.branch3_relu2(a)
        a = self.pool(a).squeeze(2)
        b = self.branch5_conv1(x)
        b = self.branch5_bn1(b)
        b = self.branch5_relu1(b)
        b = self.branch5_conv2(b)
        b = self.branch5_bn2(b)
        b = self.branch5_relu2(b)
        b = self.pool(b).squeeze(2)
        c = self.branch7_conv1(x)
        c = self.branch7_bn1(c)
        c = self.branch7_relu1(c)
        c = self.branch7_conv2(c)
        c = self.branch7_bn2(c)
        c = self.branch7_relu2(c)
        c = self.pool(c).squeeze(2)
        y = torch.cat((a, b, c), dim=1)
        y = self.fc1(y)
        y = self.fc1_relu(y)
        y = self.fc1_dropout(y)
        y = self.fc2(y)
        return y