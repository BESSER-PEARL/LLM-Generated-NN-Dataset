'''
Generate a complete PyTorch NN code that satisfies the following four Requirements: 

Architecture: RNN-Simple 
Learning Task: Regression
Input Type + Scale: Text, seq_length: Large >2000 and vocab_size: Large >100k 
Complexity: Wide-Deep: Number of RNN-Simple layers at least 2, embedding_dim of the Embedding layer at least min(128, upper_bound_of_vocab_size).


- Provide complete PyTorch code for the model only.
- All layer parameters must be fixed values inside the network, with no variables or external constants used.
- Avoid creating excessively large layers that would cause unreasonable memory usage.
- The model must only use standard layers directly, without defining helper functions or custom block classes.
- Output only the code, with no comments.
'''
import torch
import torch.nn as nn

class TextRNNRegression(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings=120000, embedding_dim=128, padding_idx=0)
        self.input_dropout = nn.Dropout(p=0.1)
        self.rnn = nn.RNN(input_size=128, hidden_size=256, num_layers=2, nonlinearity='tanh', batch_first=True, dropout=0.2, bidirectional=False)
        self.post_rnn_dropout = nn.Dropout(p=0.2)
        self.fc1 = nn.Linear(256, 128)
        self.relu = nn.ReLU()
        self.fc_dropout = nn.Dropout(p=0.2)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        x = self.embedding(x)
        x = self.input_dropout(x)
        x, _ = self.rnn(x)
        x = x[:, -1, :]
        x = self.post_rnn_dropout(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc_dropout(x)
        x = self.fc2(x)
        return x