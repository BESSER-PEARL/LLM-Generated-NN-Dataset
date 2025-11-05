'''
Generate a complete PyTorch NN code that satisfies the following four Requirements: 

Architecture: RNN-GRU 
Learning Task: Classification-multiclass
Input Type + Scale: Time series, seq_length: Small <50 and features: Large >2000 
Complexity: Wide: hidden_size of first RNN-GRU layer at least min(128, upper_bound_of_features).


- Provide complete PyTorch code for the model only.
- All layer parameters must be fixed values inside the network, with no variables or external constants used.
- Avoid creating excessively large layers that would cause unreasonable memory usage.
- The model must only use standard layers directly, without defining helper functions or custom block classes.
- Output only the code, with no comments.
'''
import torch
import torch.nn as nn

class GRUMulticlassClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.gru1 = nn.GRU(input_size=2048, hidden_size=256, num_layers=1, batch_first=True, bidirectional=False)
        self.dropout1 = nn.Dropout(p=0.2)
        self.gru2 = nn.GRU(input_size=256, hidden_size=128, num_layers=1, batch_first=True, bidirectional=False)
        self.dropout2 = nn.Dropout(p=0.2)
        self.fc1 = nn.Linear(128, 64)
        self.relu = nn.ReLU()
        self.dropout3 = nn.Dropout(p=0.2)
        self.fc_out = nn.Linear(64, 10)

    def forward(self, x):
        out1, _ = self.gru1(x)
        out1 = self.dropout1(out1)
        out2, h2 = self.gru2(out1)
        h = h2[-1]
        h = self.dropout2(h)
        h = self.fc1(h)
        h = self.relu(h)
        h = self.dropout3(h)
        logits = self.fc_out(h)
        return logits