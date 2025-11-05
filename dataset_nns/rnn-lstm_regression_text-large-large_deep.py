'''
Generate a complete PyTorch NN code that satisfies the following four Requirements: 

Architecture: RNN-LSTM 
Learning Task: Regression
Input Type + Scale: Text, seq_length: Large >2000 and vocab_size: Large >100k 
Complexity: Deep: Number of RNN-LSTM layers at least 2.


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
        self.embedding = nn.Embedding(num_embeddings=120000, embedding_dim=64)
        self.lstm = nn.LSTM(input_size=64, hidden_size=128, num_layers=3, batch_first=True, dropout=0.2, bidirectional=False)
        self.dropout = nn.Dropout(p=0.2)
        self.fc1 = nn.Linear(in_features=128, out_features=64)
        self.activation = nn.ReLU()
        self.fc2 = nn.Linear(in_features=64, out_features=1)

    def forward(self, x):
        x = self.embedding(x)
        output, (h_n, c_n) = self.lstm(x)
        x = h_n[-1]
        x = self.dropout(x)
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        return x