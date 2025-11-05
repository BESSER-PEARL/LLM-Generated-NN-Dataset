'''
Generate a complete PyTorch NN code that satisfies the following four Requirements: 

Architecture: RNN-LSTM 
Learning Task: Classification-multiclass
Input Type + Scale: Text, seq_length: Small <50 and vocab_size: Large >100k 
Complexity: Wide-Deep: Number of RNN-LSTM layers at least 2, embedding_dim of the Embedding layer at least min(128, upper_bound_of_vocab_size).


- Provide complete PyTorch code for the model only.
- All layer parameters must be fixed values inside the network, with no variables or external constants used.
- Avoid creating excessively large layers that would cause unreasonable memory usage.
- The model must only use standard layers directly, without defining helper functions or custom block classes.
- Output only the code, with no comments.
'''
import torch
import torch.nn as nn

class RNNLSTMClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings=120000, embedding_dim=128, padding_idx=0)
        self.lstm = nn.LSTM(input_size=128, hidden_size=256, num_layers=3, batch_first=True, dropout=0.3, bidirectional=True)
        self.fc1 = nn.Linear(1024, 256)
        self.dropout = nn.Dropout(0.4)
        self.relu = nn.ReLU()
        self.fc_out = nn.Linear(256, 20)

    def forward(self, x):
        x = self.embedding(x)
        x, _ = self.lstm(x)
        mean_pool = x.mean(dim=1)
        max_pool, _ = torch.max(x, dim=1)
        x = torch.cat([mean_pool, max_pool], dim=1)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.fc_out(x)
        return x