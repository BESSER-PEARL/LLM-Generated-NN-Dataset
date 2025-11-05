'''
Generate a complete PyTorch NN code that satisfies the following four Requirements: 

Architecture: RNN-LSTM 
Learning Task: Classification-binary
Input Type + Scale: Time series, seq_length: Large >2000 and features: Large >2000 
Complexity: Simple: Number of RNN-LSTM layers up to 2, hidden_size of first RNN-LSTM layer up to min(128, upper_bound_of_features).


- Provide complete PyTorch code for the model only.
- All layer parameters must be fixed values inside the network, with no variables or external constants used.
- Avoid creating excessively large layers that would cause unreasonable memory usage.
- The model must only use standard layers directly, without defining helper functions or custom block classes.
- Output only the code, with no comments.
'''
import torch
import torch.nn as nn

class RNNLSTMBinaryClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(input_size=2048, hidden_size=128, num_layers=2, batch_first=True)
        self.dropout = nn.Dropout(p=0.1)
        self.fc = nn.Linear(128, 1)
    def forward(self, x):
        _, (hn, _) = self.lstm(x)
        x = hn[-1]
        x = self.dropout(x)
        x = self.fc(x)
        return x