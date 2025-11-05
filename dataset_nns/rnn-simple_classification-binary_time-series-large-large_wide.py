'''
Generate a complete PyTorch NN code that satisfies the following four Requirements: 

Architecture: RNN-Simple 
Learning Task: Classification-binary
Input Type + Scale: Time series, seq_length: Large >2000 and features: Large >2000 
Complexity: Wide: hidden_size of first RNN-Simple layer at least min(128, upper_bound_of_features).


- Provide complete PyTorch code for the model only.
- All layer parameters must be fixed values inside the network, with no variables or external constants used.
- Avoid creating excessively large layers that would cause unreasonable memory usage.
- The model must only use standard layers directly, without defining helper functions or custom block classes.
- Output only the code, with no comments.
'''
import torch
import torch.nn as nn

class RNNBinaryClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.rnn = nn.RNN(input_size=2048, hidden_size=128, num_layers=1, nonlinearity='tanh', batch_first=True, bidirectional=False)
        self.dropout = nn.Dropout(p=0.2)
        self.fc = nn.Linear(128, 1)
    def forward(self, x):
        out, hn = self.rnn(x)
        h_last = hn[-1]
        h_last = self.dropout(h_last)
        logits = self.fc(h_last)
        return logits