'''
Generate a complete PyTorch NN code that satisfies the following four Requirements: 

Architecture: RNN-LSTM 
Learning Task: Classification-binary
Input Type + Scale: Text, seq_length: Small <50 and vocab_size: Small <1k 
Complexity: Wide-Deep: Number of RNN-LSTM layers at least 2, embedding_dim of the Embedding layer at least min(128, upper_bound_of_vocab_size).


- Provide complete PyTorch code for the model only.
- All layer parameters must be fixed values inside the network, with no variables or external constants used.
- Avoid creating excessively large layers that would cause unreasonable memory usage.
- The model must only use standard layers directly, without defining helper functions or custom block classes.
- Output only the code, with no comments.
'''
import torch
import torch.nn as nn

class TextRNNLSTMBinaryClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(800, 256, padding_idx=0)
        self.lstm1 = nn.LSTM(256, 128, num_layers=1, batch_first=True, dropout=0.0)
        self.lstm2 = nn.LSTM(128, 128, num_layers=1, batch_first=True, dropout=0.0)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(128, 1)

    def forward(self, x):
        x = self.embedding(x)
        x, _ = self.lstm1(x)
        x, (h, _) = self.lstm2(x)
        x = h[-1]
        x = self.dropout(x)
        x = self.fc(x)
        return x