'''
Generate a complete PyTorch NN code that satisfies the following four Requirements: 

Architecture: RNN-LSTM 
Learning Task: Regression
Input Type + Scale: Text, seq_length: Large >2000 and vocab_size: Small <1k 
Complexity: Wide-Deep: Number of RNN-LSTM layers at least 2, embedding_dim of the Embedding layer at least min(128, upper_bound_of_vocab_size).


- Provide complete PyTorch code for the model only.
- All layer parameters must be fixed values inside the network, with no variables or external constants used.
- Avoid creating excessively large layers that would cause unreasonable memory usage.
- The model must only use standard layers directly, without defining helper functions or custom block classes.
- Output only the code, with no comments.
'''
import torch
import torch.nn as nn

class TextLSTMRegressor(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings=800, embedding_dim=128)
        self.lstm = nn.LSTM(input_size=128, hidden_size=256, num_layers=2, batch_first=True, dropout=0.1, bidirectional=False)
        self.dropout = nn.Dropout(0.2)
        self.fc1 = nn.Linear(256, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        x = self.embedding(x)
        out, (hn, cn) = self.lstm(x)
        h = hn[-1]
        h = self.dropout(h)
        h = self.relu(self.fc1(h))
        h = self.dropout(h)
        y = self.fc2(h)
        return y