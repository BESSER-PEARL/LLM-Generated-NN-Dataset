'''
Generate a complete PyTorch NN code that satisfies the following four Requirements: 

Architecture: CNN-1D 
Learning Task: Representation-learning
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

class TextCNN1DRepresentation(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings=800, embedding_dim=64, padding_idx=0)
        self.conv1 = nn.Conv1d(64, 96, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(96)
        self.conv2 = nn.Conv1d(96, 128, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(128)
        self.conv3 = nn.Conv1d(128, 128, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm1d(128)
        self.act = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=0.1)
        self.pool = nn.AdaptiveAvgPool1d(output_size=1)
        self.proj = nn.Linear(128, 128)
        self.out_norm = nn.LayerNorm(128)

    def forward(self, x):
        x = self.embedding(x)
        x = x.transpose(1, 2)
        x = self.bn1(self.conv1(x))
        x = self.act(x)
        x = self.dropout(x)
        x = self.bn2(self.conv2(x))
        x = self.act(x)
        x = self.dropout(x)
        x = self.bn3(self.conv3(x))
        x = self.act(x)
        x = self.pool(x).squeeze(-1)
        x = self.proj(x)
        x = self.out_norm(x)
        return x