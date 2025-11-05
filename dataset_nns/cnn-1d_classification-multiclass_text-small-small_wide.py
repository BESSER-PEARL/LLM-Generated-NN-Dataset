'''
Generate a complete PyTorch NN code that satisfies the following four Requirements: 

Architecture: CNN-1D 
Learning Task: Classification-multiclass
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
        super(TextCNN1D, self).__init__()
        self.embedding = nn.Embedding(800, 128, padding_idx=0)
        self.dropout_emb = nn.Dropout(0.2)
        self.conv1 = nn.Conv1d(128, 128, 3, padding=1)
        self.conv2 = nn.Conv1d(128, 128, 5, padding=2)
        self.conv3 = nn.Conv1d(128, 128, 7, padding=3)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(384, 10)

    def forward(self, x):
        x = self.embedding(x)
        x = self.dropout_emb(x)
        x = x.transpose(1, 2)
        x1 = self.pool(self.relu(self.conv1(x))).squeeze(-1)
        x2 = self.pool(self.relu(self.conv2(x))).squeeze(-1)
        x3 = self.pool(self.relu(self.conv3(x))).squeeze(-1)
        out = torch.cat((x1, x2, x3), dim=1)
        out = self.dropout(out)
        out = self.fc(out)
        return out