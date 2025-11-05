'''
Generate a complete PyTorch NN code that satisfies the following four Requirements: 

Architecture: CNN-1D 
Learning Task: Classification-multiclass
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

class TextCNN1D(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings=120000, embedding_dim=128, padding_idx=0)
        self.dropout_emb = nn.Dropout(p=0.2)
        self.conv3 = nn.Conv1d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=0, dilation=1, groups=1, bias=True)
        self.conv4 = nn.Conv1d(in_channels=128, out_channels=64, kernel_size=4, stride=1, padding=0, dilation=1, groups=1, bias=True)
        self.conv5 = nn.Conv1d(in_channels=128, out_channels=64, kernel_size=5, stride=1, padding=0, dilation=1, groups=1, bias=True)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.AdaptiveMaxPool1d(output_size=1)
        self.dropout_fc = nn.Dropout(p=0.5)
        self.fc = nn.Linear(in_features=192, out_features=10, bias=True)

    def forward(self, x):
        x = self.embedding(x)
        x = self.dropout_emb(x)
        x = x.transpose(1, 2)
        x3 = self.pool(self.relu(self.conv3(x))).squeeze(-1)
        x4 = self.pool(self.relu(self.conv4(x))).squeeze(-1)
        x5 = self.pool(self.relu(self.conv5(x))).squeeze(-1)
        x = torch.cat([x3, x4, x5], dim=1)
        x = self.dropout_fc(x)
        x = self.fc(x)
        return x