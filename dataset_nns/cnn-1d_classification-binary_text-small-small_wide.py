'''
Generate a complete PyTorch NN code that satisfies the following four Requirements: 

Architecture: CNN-1D 
Learning Task: Classification-binary
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

class TextCNN1D(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings=800, embedding_dim=128, padding_idx=0)
        self.conv3 = nn.Conv1d(in_channels=128, out_channels=96, kernel_size=3)
        self.conv4 = nn.Conv1d(in_channels=128, out_channels=96, kernel_size=4)
        self.conv5 = nn.Conv1d(in_channels=128, out_channels=96, kernel_size=5)
        self.relu = nn.ReLU()
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.dropout1 = nn.Dropout(0.3)
        self.fc1 = nn.Linear(288, 64)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x):
        x = self.embedding(x)
        x = x.transpose(1, 2)
        x1 = self.pool(self.relu(self.conv3(x))).squeeze(-1)
        x2 = self.pool(self.relu(self.conv4(x))).squeeze(-1)
        x3 = self.pool(self.relu(self.conv5(x))).squeeze(-1)
        x = torch.cat([x1, x2, x3], dim=1)
        x = self.dropout1(x)
        x = self.fc1(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        return x