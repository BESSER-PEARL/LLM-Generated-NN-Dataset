'''
Generate a complete PyTorch NN code that satisfies the following four Requirements: 

Architecture: RNN-LSTM 
Learning Task: Regression
Input Type + Scale: Text, seq_length: Small <50 and vocab_size: Large >100k 
Complexity: Deep: Number of RNN-LSTM layers at least 2.


- Provide complete PyTorch code for the model only.
- All layer parameters must be fixed values inside the network, with no variables or external constants used.
- Avoid creating excessively large layers that would cause unreasonable memory usage.
- The model must only use standard layers directly, without defining helper functions or custom block classes.
- Output only the code, with no comments.
'''
import torch
import torch.nn as nn

class LSTMTextRegressor(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings=120000, embedding_dim=64, padding_idx=0)
        self.lstm = nn.LSTM(input_size=64, hidden_size=128, num_layers=2, dropout=0.2, batch_first=True)
        self.dropout = nn.Dropout(p=0.2)
        self.fc = nn.Linear(128, 1)

    def forward(self, x):
        x = self.embedding(x)
        output, _ = self.lstm(x)
        x = output[:, -1, :]
        x = self.dropout(x)
        x = self.fc(x)
        return x