'''
Generate a complete PyTorch NN code that satisfies the following four Requirements: 

Architecture: CNN-1D 
Learning Task: Regression
Input Type + Scale: Text, seq_length: Small <50 and vocab_size: Large >100k 
Complexity: Wide: embedding_dim of the Embedding layer at least min(128, upper_bound_of_vocab_size).


- Provide complete PyTorch code for the model only.
- All layer parameters must be fixed values inside the network, with no variables or external constants used.
- Avoid creating excessively large layers that would cause unreasonable memory usage.
- The model must only use standard layers directly, without defining helper functions or custom block classes.
- Output only the code, with no comments.
'''
import torch
import torch.nn as nn

class TextCNN1DRegression(nn.Module):
    def __init__(self):
        super(TextCNN1DRegression, self).__init__()
        self.embedding = nn.Embedding(num_embeddings=200000, embedding_dim=128, padding_idx=0)
        self.conv1 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=3, padding=1, bias=True)
        self.relu1 = nn.ReLU()
        self.bn1 = nn.BatchNorm1d(num_features=128)
        self.conv2 = nn.Conv1d(in_channels=128, out_channels=64, kernel_size=5, padding=2, bias=True)
        self.relu2 = nn.ReLU()
        self.bn2 = nn.BatchNorm1d(num_features=64)
        self.pool = nn.AdaptiveMaxPool1d(output_size=1)
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(p=0.1)
        self.fc1 = nn.Linear(in_features=64, out_features=32, bias=True)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(in_features=32, out_features=1, bias=True)

    def forward(self, x):
        x = self.embedding(x)
        x = x.transpose(1, 2)
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.bn1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.bn2(x)
        x = self.pool(x)
        x = self.flatten(x)
        x = self.dropout(x)
        x = self.fc1(x)
        x = self.relu3(x)
        x = self.fc2(x)
        return x