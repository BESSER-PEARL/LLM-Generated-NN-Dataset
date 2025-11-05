'''
Generate a complete PyTorch NN code that satisfies the following four Requirements: 

Architecture: RNN-LSTM 
Learning Task: Classification-multiclass
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

class TextLSTMClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(800, 128, padding_idx=0)
        self.dropout_embed = nn.Dropout(0.2)
        self.lstm = nn.LSTM(128, 160, num_layers=2, batch_first=True, bidirectional=True, dropout=0.25)
        self.dropout_lstm = nn.Dropout(0.3)
        self.fc1 = nn.Linear(320, 128)
        self.relu = nn.ReLU()
        self.dropout_fc = nn.Dropout(0.3)
        self.fc2 = nn.Linear(128, 8)

    def forward(self, x):
        x = self.embedding(x)
        x = self.dropout_embed(x)
        output, (hn, cn) = self.lstm(x)
        h = torch.cat((hn[-2], hn[-1]), dim=1)
        h = self.dropout_lstm(h)
        h = self.fc1(h)
        h = self.relu(h)
        h = self.dropout_fc(h)
        logits = self.fc2(h)
        return logits