'''
Generate a complete PyTorch NN code that satisfies the following four Requirements: 

Architecture: RNN-Simple 
Learning Task: Classification-multiclass
Input Type + Scale: Time series, seq_length: Large >2000 and features: Large >2000 
Complexity: Deep: Number of RNN-Simple layers at least 2.


- Provide complete PyTorch code for the model only.
- All layer parameters must be fixed values inside the network, with no variables or external constants used.
- Avoid creating excessively large layers that would cause unreasonable memory usage.
- The model must only use standard layers directly, without defining helper functions or custom block classes.
- Output only the code, with no comments.
'''
import torch
import torch.nn as nn

class LargeSeqRNNClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.rnn1 = nn.RNN(input_size=2048, hidden_size=64, num_layers=1, nonlinearity='tanh', batch_first=True, dropout=0.0, bidirectional=False)
        self.rnn2 = nn.RNN(input_size=64, hidden_size=64, num_layers=2, nonlinearity='tanh', batch_first=True, dropout=0.1, bidirectional=False)
        self.dropout = nn.Dropout(p=0.2)
        self.fc = nn.Linear(64, 10)

    def forward(self, x):
        x, _ = self.rnn1(x)
        x, _ = self.rnn2(x)
        x = x[:, -1, :]
        x = self.dropout(x)
        x = self.fc(x)
        return x