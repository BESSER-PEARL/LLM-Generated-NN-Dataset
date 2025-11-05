'''
Generate a complete PyTorch NN code that satisfies the following four Requirements: 

Architecture: RNN-GRU 
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

class TextGRUBinaryClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(900, 128, padding_idx=0)
        self.embed_dropout = nn.Dropout(0.2)
        self.gru = nn.GRU(input_size=128, hidden_size=128, num_layers=1, batch_first=True, bidirectional=True)
        self.fc1 = nn.Linear(256, 64)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(64, 1)
    def forward(self, x):
        x = self.embedding(x)
        x = self.embed_dropout(x)
        _, hn = self.gru(x)
        hn = torch.cat((hn[-2], hn[-1]), dim=1)
        x = self.fc1(hn)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x