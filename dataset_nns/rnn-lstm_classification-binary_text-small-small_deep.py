'''
Generate a complete PyTorch NN code that satisfies the following four Requirements: 

Architecture: RNN-LSTM 
Learning Task: Classification-binary
Input Type + Scale: Text, seq_length: Small <50 and vocab_size: Small <1k 
Complexity: Deep: Number of RNN-LSTM layers at least 2.


- Provide complete PyTorch code for the model only.
- All layer parameters must be fixed values inside the network, with no variables or external constants used.
- Avoid creating excessively large layers that would cause unreasonable memory usage.
- The model must only use standard layers directly, without defining helper functions or custom block classes.
- Output only the code, with no comments.
'''
import torch
import torch.nn as nn

class BinaryTextLSTM(nn.Module):
    def __init__(self):
        super(BinaryTextLSTM, self).__init__()
        self.embedding = nn.Embedding(num_embeddings=800, embedding_dim=64, padding_idx=0)
        self.lstm = nn.LSTM(input_size=64, hidden_size=128, num_layers=2, batch_first=True, dropout=0.2, bidirectional=False)
        self.dropout = nn.Dropout(p=0.3)
        self.fc = nn.Linear(128, 1)
    def forward(self, x):
        x = self.embedding(x)
        _, (h_n, _) = self.lstm(x)
        h = self.dropout(h_n[-1])
        return self.fc(h)