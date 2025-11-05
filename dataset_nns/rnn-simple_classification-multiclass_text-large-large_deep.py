'''
Generate a complete PyTorch NN code that satisfies the following four Requirements: 

Architecture: RNN-Simple 
Learning Task: Classification-multiclass
Input Type + Scale: Text, seq_length: Large >2000 and vocab_size: Large >100k 
Complexity: Deep: Number of RNN-Simple layers at least 2.


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
        self.embedding = nn.Embedding(num_embeddings=110000, embedding_dim=64, padding_idx=0)
        self.drop_in = nn.Dropout(p=0.1)
        self.rnn1 = nn.RNN(input_size=64, hidden_size=128, num_layers=1, nonlinearity='tanh', batch_first=True, bidirectional=False)
        self.drop_mid = nn.Dropout(p=0.1)
        self.rnn2 = nn.RNN(input_size=128, hidden_size=128, num_layers=1, nonlinearity='tanh', batch_first=True, bidirectional=False)
        self.drop_out = nn.Dropout(p=0.1)
        self.fc = nn.Linear(in_features=128, out_features=20)

    def forward(self, x):
        x = self.embedding(x)
        x = self.drop_in(x)
        x, _ = self.rnn1(x)
        x = self.drop_mid(x)
        x, _ = self.rnn2(x)
        x = x[:, -1, :]
        x = self.drop_out(x)
        x = self.fc(x)
        return x