'''
Generate a complete PyTorch NN code that satisfies the following four Requirements: 

Architecture: RNN-LSTM 
Learning Task: Classification-multiclass
Input Type + Scale: Text, seq_length: Large >2000 and vocab_size: Large >100k 
Complexity: Simple: Number of RNN-LSTM layers up to 2, embedding_dim of the Embedding layer up to min(128, upper_bound_of_vocab_size).


- Provide complete PyTorch code for the model only.
- All layer parameters must be fixed values inside the network, with no variables or external constants used.
- Avoid creating excessively large layers that would cause unreasonable memory usage.
- The model must only use standard layers directly, without defining helper functions or custom block classes.
- Output only the code, with no comments.
'''
import torch
import torch.nn as nn

class RNNLSTMTextClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(120000, 128, padding_idx=0)
        self.lstm = nn.LSTM(input_size=128, hidden_size=128, num_layers=2, batch_first=True, dropout=0.1, bidirectional=False)
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(128, 20)

    def forward(self, x):
        x = self.embedding(x)
        output, (h_n, c_n) = self.lstm(x)
        h_last = h_n[-1]
        h_last = self.dropout(h_last)
        logits = self.fc(h_last)
        return logits