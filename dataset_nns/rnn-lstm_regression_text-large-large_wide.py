'''
Generate a complete PyTorch NN code that satisfies the following four Requirements: 

Architecture: RNN-LSTM 
Learning Task: Regression
Input Type + Scale: Text, seq_length: Large >2000 and vocab_size: Large >100k 
Complexity: Wide: embedding_dim of the Embedding layer at least min(128, upper_bound_of_vocab_size).


- Provide complete PyTorch code for the model only.
- All layer parameters must be fixed values inside the network, with no variables or external constants used.
- Avoid creating excessively large layers that would cause unreasonable memory usage.
- The model must only use standard layers directly, without defining helper functions or custom block classes.
- Output only the code, with no comments.
'''
import torch
import torch.nn as nn

class RNNLSTMRegressor(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(100001, 128, padding_idx=0)
        self.lstm = nn.LSTM(input_size=128, hidden_size=64, num_layers=1, batch_first=True, dropout=0.0, bidirectional=False)
        self.dropout = nn.Dropout(0.1)
        self.fc1 = nn.Linear(64, 32)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(32, 1)

    def forward(self, x):
        x = self.embedding(x)
        output, (hn, cn) = self.lstm(x)
        h = hn[-1]
        h = self.dropout(h)
        h = self.fc1(h)
        h = self.relu(h)
        y = self.fc2(h)
        return y