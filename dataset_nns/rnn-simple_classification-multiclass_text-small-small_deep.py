'''
Generate a complete PyTorch NN code that satisfies the following four Requirements: 

Architecture: RNN-Simple 
Learning Task: Classification-multiclass
Input Type + Scale: Text, seq_length: Small <50 and vocab_size: Small <1k 
Complexity: Deep: Number of RNN-Simple layers at least 2.


- Provide complete PyTorch code for the model only.
- All layer parameters must be fixed values inside the network, with no variables or external constants used.
- Avoid creating excessively large layers that would cause unreasonable memory usage.
- The model must only use standard layers directly, without defining helper functions or custom block classes.
- Output only the code, with no comments.
'''
import torch
import torch.nn as nn

class TextRNNClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(500, 64, padding_idx=0)
        self.dropout_emb = nn.Dropout(0.2)
        self.rnn = nn.RNN(input_size=64, hidden_size=64, num_layers=2, nonlinearity='tanh', batch_first=True, dropout=0.2, bidirectional=False)
        self.dropout_fc = nn.Dropout(0.2)
        self.fc = nn.Linear(64, 8)
    def forward(self, x):
        x = self.embedding(x)
        x = self.dropout_emb(x)
        output, hidden = self.rnn(x)
        h_last = hidden[-1]
        h_last = self.dropout_fc(h_last)
        logits = self.fc(h_last)
        return logits