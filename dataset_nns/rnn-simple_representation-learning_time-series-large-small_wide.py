'''
Generate a complete PyTorch NN code that satisfies the following four Requirements: 

Architecture: RNN-Simple 
Learning Task: Representation-learning
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

class RnnSimpleRepresentationModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.rnn1 = nn.RNN(input_size=32, hidden_size=64, num_layers=1, nonlinearity="tanh", batch_first=True, dropout=0.0, bidirectional=False)
        self.rnn2 = nn.RNN(input_size=64, hidden_size=64, num_layers=1, nonlinearity="tanh", batch_first=True, dropout=0.0, bidirectional=False)
        self.fc1 = nn.Linear(64, 64)
        self.act = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(64, 32)

    def forward(self, x):
        out1, h1 = self.rnn1(x)
        out2, h2 = self.rnn2(out1)
        rep = h2[-1]
        z = self.fc2(self.act(self.fc1(rep)))
        return z