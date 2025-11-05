'''
Generate a complete PyTorch NN code that satisfies the following four Requirements: 

Architecture: RNN-Simple 
Learning Task: Classification-multiclass
Input Type + Scale: Time series, seq_length: Large >2000 and features: Small <50 
Complexity: Wide: hidden_size of first RNN-Simple layer at least min(128, upper_bound_of_features).


- Provide complete PyTorch code for the model only.
- All layer parameters must be fixed values inside the network, with no variables or external constants used.
- Avoid creating excessively large layers that would cause unreasonable memory usage.
- The model must only use standard layers directly, without defining helper functions or custom block classes.
- Output only the code, with no comments.
'''
import torch
import torch.nn as nn

class RNNClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.rnn1 = nn.RNN(input_size=32, hidden_size=128, num_layers=1, batch_first=True, nonlinearity='tanh', bias=True)
        self.rnn2 = nn.RNN(input_size=128, hidden_size=64, num_layers=1, batch_first=True, nonlinearity='tanh', bias=True)
        self.dropout = nn.Dropout(p=0.3)
        self.fc = nn.Linear(64, 10)
    def forward(self, x):
        out1, h1 = self.rnn1(x)
        out2, h2 = self.rnn2(out1)
        h = h2[-1]
        h = self.dropout(h)
        logits = self.fc(h)
        return logits