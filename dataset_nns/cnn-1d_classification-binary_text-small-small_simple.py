'''
Generate a complete PyTorch NN code that satisfies the following four Requirements: 

Architecture: CNN-1D 
Learning Task: Classification-binary
Input Type + Scale: Text, seq_length: Small <50 and vocab_size: Small <1k 
Complexity: Simple: Number of CNN-1D layers up to 4, embedding_dim of the Embedding layer up to min(128, upper_bound_of_vocab_size).


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
        self.embedding = nn.Embedding(num_embeddings=800, embedding_dim=64, padding_idx=0)
        self.dropout = nn.Dropout(p=0.2)
        self.conv1 = nn.Conv1d(in_channels=64, out_channels=96, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(num_features=96)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(in_channels=96, out_channels=128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(num_features=128)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(num_features=128)
        self.relu3 = nn.ReLU(inplace=True)
        self.pool = nn.AdaptiveMaxPool1d(output_size=1)
        self.fc = nn.Linear(in_features=128, out_features=1)

    def forward(self, x):
        x = self.embedding(x)
        x = self.dropout(x)
        x = x.transpose(1, 2)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        x = self.pool(x).squeeze(-1)
        x = self.fc(x)
        return x