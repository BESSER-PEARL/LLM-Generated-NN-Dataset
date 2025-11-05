'''
Generate a complete PyTorch NN code that satisfies the following four Requirements: 

Architecture: RNN-Simple 
Learning Task: Classification-binary
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

class BinaryRNNClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.rnn = nn.RNN(input_size=2048, hidden_size=64, num_layers=3, nonlinearity='tanh', batch_first=True, dropout=0.2, bidirectional=False)
        self.dropout = nn.Dropout(p=0.3)
        self.fc = nn.Linear(64, 1)

    def forward(self, x):
        _, hn = self.rnn(x)
        h = hn[-1]
        h = self.dropout(h)
        out = self.fc(h)
        return out