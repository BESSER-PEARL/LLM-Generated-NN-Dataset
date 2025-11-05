'''
Generate a complete PyTorch NN code that satisfies the following four Requirements: 

Architecture: CNN-1D 
Learning Task: Classification-multiclass
Input Type + Scale: Text, seq_length: Small <50 and vocab_size: Small <1k 
Complexity: Deep: Number of CNN-1D layers at least 4.


- Provide complete PyTorch code for the model only.
- All layer parameters must be fixed values inside the network, with no variables or external constants used.
- Avoid creating excessively large layers that would cause unreasonable memory usage.
- The model must only use standard layers directly, without defining helper functions or custom block classes.
- Output only the code, with no comments.
'''
import torch
import torch.nn as nn

class DeepTextCNN1D(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings=800, embedding_dim=64, padding_idx=0)
        self.dropout_embed = nn.Dropout(p=0.2)
        self.conv1 = nn.Conv1d(in_channels=64, out_channels=96, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn1 = nn.BatchNorm1d(num_features=96)
        self.conv2 = nn.Conv1d(in_channels=96, out_channels=128, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn2 = nn.BatchNorm1d(num_features=128)
        self.conv3 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn3 = nn.BatchNorm1d(num_features=128)
        self.conv4 = nn.Conv1d(in_channels=128, out_channels=160, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn4 = nn.BatchNorm1d(num_features=160)
        self.conv5 = nn.Conv1d(in_channels=160, out_channels=160, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn5 = nn.BatchNorm1d(num_features=160)
        self.pool = nn.AdaptiveMaxPool1d(output_size=1)
        self.classifier = nn.Sequential(
            nn.Linear(in_features=160, out_features=128, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(in_features=128, out_features=7, bias=True)
        )

    def forward(self, x):
        x = self.embedding(x)
        x = self.dropout_embed(x)
        x = x.permute(0, 2, 1)
        x = torch.relu(self.bn1(self.conv1(x)))
        x = torch.relu(self.bn2(self.conv2(x)))
        x = torch.relu(self.bn3(self.conv3(x)))
        x = torch.relu(self.bn4(self.conv4(x)))
        x = torch.relu(self.bn5(self.conv5(x)))
        x = self.pool(x).squeeze(-1)
        x = self.classifier(x)
        return x