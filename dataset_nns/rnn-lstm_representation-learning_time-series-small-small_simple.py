'''
Generate a complete PyTorch NN code that satisfies the following four Requirements: 

Architecture: RNN-LSTM 
Learning Task: Representation-learning
Input Type + Scale: Time series, seq_length: Small <50 and features: Small <50 
Complexity: Simple: Number of RNN-LSTM layers up to 2, hidden_size of first RNN-LSTM layer up to min(128, upper_bound_of_features).


- Provide complete PyTorch code for the model only.
- All layer parameters must be fixed values inside the network, with no variables or external constants used.
- Avoid creating excessively large layers that would cause unreasonable memory usage.
- The model must only use standard layers directly, without defining helper functions or custom block classes.
- Output only the code, with no comments.
'''
import torch
import torch.nn as nn

class LSTMRepresentationModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(input_size=16, hidden_size=16, num_layers=2, batch_first=True)
        self.projection = nn.Linear(16, 16)
        self.normalization = nn.LayerNorm(16)
        self.activation = nn.Tanh()

    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        z = h_n[-1]
        z = self.projection(z)
        z = self.normalization(z)
        z = self.activation(z)
        return z