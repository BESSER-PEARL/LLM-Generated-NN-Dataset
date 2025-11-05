'''
Generate a complete PyTorch NN code that satisfies the following four Requirements: 

Architecture: RNN-Simple 
Learning Task: Classification-multiclass
Input Type + Scale: Time series, seq_length: Small <50 and features: Large >2000 
Complexity: Deep: Number of RNN-Simple layers at least 2.


- Provide complete PyTorch code for the model only.
- All layer parameters must be fixed values inside the network, with no variables or external constants used.
- Avoid creating excessively large layers that would cause unreasonable memory usage.
- The model must only use standard layers directly, without defining helper functions or custom block classes.
- Output only the code, with no comments.
'''
import torch
import torch.nn as nn

class TimeSeriesRNNClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.rnn1 = nn.RNN(input_size=2048, hidden_size=128, num_layers=1, nonlinearity='tanh', batch_first=True, bidirectional=False, dropout=0.0)
        self.dropout1 = nn.Dropout(p=0.2)
        self.rnn2 = nn.RNN(input_size=128, hidden_size=128, num_layers=1, nonlinearity='tanh', batch_first=True, bidirectional=False, dropout=0.0)
        self.dropout2 = nn.Dropout(p=0.2)
        self.fc = nn.Linear(128, 10)

    def forward(self, x):
        out1, h1 = self.rnn1(x)
        out1 = self.dropout1(out1)
        out2, h2 = self.rnn2(out1)
        out2 = self.dropout2(h2[-1])
        logits = self.fc(out2)
        return logits