'''
Generate a complete PyTorch NN code that satisfies the following four Requirements: 

Architecture: CNN-1D 
Learning Task: Classification-binary
Input Type + Scale: Text, seq_length: Large >2000 and vocab_size: Small <1k 
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
        self.embedding = nn.Embedding(num_embeddings=800, embedding_dim=128, padding_idx=0)
        self.dropout_embed = nn.Dropout(p=0.2)
        self.conv3 = nn.Conv1d(in_channels=128, out_channels=64, kernel_size=3, padding=1)
        self.conv5 = nn.Conv1d(in_channels=128, out_channels=64, kernel_size=5, padding=2)
        self.conv7 = nn.Conv1d(in_channels=128, out_channels=64, kernel_size=7, padding=3)
        self.act = nn.ReLU()
        self.pool = nn.AdaptiveMaxPool1d(output_size=1)
        self.fc1 = nn.Linear(in_features=192, out_features=64)
        self.dropout_fc = nn.Dropout(p=0.5)
        self.fc_out = nn.Linear(in_features=64, out_features=1)

    def forward(self, x):
        x = self.embedding(x)
        x = self.dropout_embed(x)
        x = x.transpose(1, 2)
        x3 = self.pool(self.act(self.conv3(x))).squeeze(-1)
        x5 = self.pool(self.act(self.conv5(x))).squeeze(-1)
        x7 = self.pool(self.act(self.conv7(x))).squeeze(-1)
        x_cat = torch.cat((x3, x5, x7), dim=1)
        h = self.act(self.fc1(x_cat))
        h = self.dropout_fc(h)
        out = self.fc_out(h)
        return out