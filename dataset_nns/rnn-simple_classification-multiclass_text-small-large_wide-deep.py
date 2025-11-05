'''
Generate a complete PyTorch NN code that satisfies the following four Requirements: 

Architecture: RNN-Simple 
Learning Task: Classification-multiclass
Input Type + Scale: Text, seq_length: Small <50 and vocab_size: Large >100k 
Complexity: Wide-Deep: Number of RNN-Simple layers at least 2, embedding_dim of the Embedding layer at least min(128, upper_bound_of_vocab_size).


- Provide complete PyTorch code for the model only.
- All layer parameters must be fixed values inside the network, with no variables or external constants used.
- Avoid creating excessively large layers that would cause unreasonable memory usage.
- The model must only use standard layers directly, without defining helper functions or custom block classes.
- Output only the code, with no comments.
'''
import torch
import torch.nn as nn

class TextRNNClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(120000, 128, padding_idx=0)
        self.rnn = nn.RNN(128, 256, num_layers=2, batch_first=True, nonlinearity='tanh', dropout=0.25, bidirectional=True)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(512, 20)

    def forward(self, x):
        x = self.embedding(x)
        output, _ = self.rnn(x)
        x = output[:, -1, :]
        x = self.dropout(x)
        x = self.fc(x)
        return x