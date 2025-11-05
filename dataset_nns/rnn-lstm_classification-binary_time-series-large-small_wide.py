'''
Generate a complete PyTorch NN code that satisfies the following four Requirements: 

Architecture: RNN-LSTM 
Learning Task: Classification-binary
Input Type + Scale: Time series, seq_length: Large >2000 and features: Small <50 
Complexity: Wide: hidden_size of first RNN-LSTM layer at least min(128, upper_bound_of_features).


- Provide complete PyTorch code for the model only.
- All layer parameters must be fixed values inside the network, with no variables or external constants used.
- Avoid creating excessively large layers that would cause unreasonable memory usage.
- The model must only use standard layers directly, without defining helper functions or custom block classes.
- Output only the code, with no comments.
'''
import torch
import torch.nn as nn

class BinaryLSTMClassifier(nn.Module):
    def __init__(self):
        super(BinaryLSTMClassifier, self).__init__()
        self.lstm1 = nn.LSTM(input_size=32, hidden_size=128, num_layers=1, batch_first=True, bidirectional=False, dropout=0.0)
        self.dropout1 = nn.Dropout(p=0.2)
        self.lstm2 = nn.LSTM(input_size=128, hidden_size=64, num_layers=1, batch_first=True, bidirectional=False, dropout=0.0)
        self.dropout2 = nn.Dropout(p=0.2)
        self.fc1 = nn.Linear(in_features=64, out_features=32)
        self.relu = nn.ReLU()
        self.dropout3 = nn.Dropout(p=0.2)
        self.fc_out = nn.Linear(in_features=32, out_features=1)

    def forward(self, x):
        out, _ = self.lstm1(x)
        out = self.dropout1(out)
        out, _ = self.lstm2(out)
        out = out[:, -1, :]
        out = self.dropout2(out)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.dropout3(out)
        out = self.fc_out(out)
        return out