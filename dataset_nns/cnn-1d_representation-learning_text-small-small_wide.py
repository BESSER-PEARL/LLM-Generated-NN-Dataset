'''
Generate a complete PyTorch NN code that satisfies the following four Requirements: 

Architecture: CNN-1D 
Learning Task: Representation-learning
Input Type + Scale: Text, seq_length: Small <50 and vocab_size: Small <1k 
Complexity: Wide: embedding_dim of the Embedding layer at least min(128, upper_bound_of_vocab_size).


- Provide complete PyTorch code for the model only.
- All layer parameters must be fixed values inside the network, with no variables or external constants used.
- Avoid creating excessively large layers that would cause unreasonable memory usage.
- The model must only use standard layers directly, without defining helper functions or custom block classes.
- Output only the code, with no comments.
'''
import torch
import torch.nn as nn

class TextCNN1DEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings=800, embedding_dim=256, padding_idx=0)
        self.dropout_embed = nn.Dropout(p=0.1)
        self.conv1 = nn.Conv1d(in_channels=256, out_channels=192, kernel_size=5, padding=2, bias=True)
        self.bn1 = nn.BatchNorm1d(num_features=192)
        self.act1 = nn.ReLU(inplace=True)
        self.dropout1 = nn.Dropout(p=0.2)
        self.conv2 = nn.Conv1d(in_channels=192, out_channels=192, kernel_size=3, padding=1, bias=True)
        self.bn2 = nn.BatchNorm1d(num_features=192)
        self.act2 = nn.ReLU(inplace=True)
        self.dropout2 = nn.Dropout(p=0.2)
        self.conv3 = nn.Conv1d(in_channels=192, out_channels=128, kernel_size=3, padding=1, bias=True)
        self.act3 = nn.ReLU(inplace=True)
        self.pool = nn.AdaptiveAvgPool1d(output_size=1)
        self.flatten = nn.Flatten()
        self.proj = nn.Linear(in_features=128, out_features=128)

    def forward(self, x):
        x = self.embedding(x)
        x = self.dropout_embed(x)
        x = x.transpose(1, 2)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.dropout1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.act2(x)
        x = self.dropout2(x)
        x = self.conv3(x)
        x = self.act3(x)
        x = self.pool(x)
        x = self.flatten(x)
        x = self.proj(x)
        return x