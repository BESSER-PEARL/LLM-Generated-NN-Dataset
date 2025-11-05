'''
Generate a complete PyTorch NN code that satisfies the following four Requirements: 

Architecture: RNN-Simple 
Learning Task: Classification-multiclass
Input Type + Scale: Text, seq_length: Small <50 and vocab_size: Large >100k 
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
        self.embedding = nn.Embedding(120000, 64, padding_idx=0)
        self.rnn = nn.RNN(input_size=64, hidden_size=128, num_layers=2, nonlinearity='tanh', batch_first=True, dropout=0.2, bidirectional=False)
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(128, 20)
    def forward(self, x):
        x = self.embedding(x)
        _, h_n = self.rnn(x)
        h = h_n[-1]
        h = self.dropout(h)
        logits = self.fc(h)
        return logits