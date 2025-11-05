'''
Generate a complete PyTorch NN code that satisfies the following four Requirements: 

Architecture: CNN-1D 
Learning Task: Classification-binary
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

class TextCNN1DBinary(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(120000, 128, padding_idx=0)
        self.conv3 = nn.Conv1d(128, 128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv1d(128, 128, kernel_size=4, padding=2)
        self.conv5 = nn.Conv1d(128, 128, kernel_size=5, padding=2)
        self.relu = nn.ReLU()
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(384, 1)

    def forward(self, x):
        x = self.embedding(x)
        x = x.transpose(1, 2)
        x3 = self.pool(self.relu(self.conv3(x))).squeeze(-1)
        x4 = self.pool(self.relu(self.conv4(x))).squeeze(-1)
        x5 = self.pool(self.relu(self.conv5(x))).squeeze(-1)
        x = torch.cat([x3, x4, x5], dim=1)
        x = self.dropout(x)
        x = self.fc(x)
        return x.squeeze(-1)