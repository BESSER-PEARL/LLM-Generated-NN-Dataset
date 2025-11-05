'''
Generate a complete PyTorch NN code that satisfies the following four Requirements: 

Architecture: RNN-Simple 
Learning Task: Classification-binary
Input Type + Scale: Text, seq_length: Small <50 and vocab_size: Small <1k 
Complexity: Deep: Number of RNN-Simple layers at least 2.


- Provide complete PyTorch code for the model only.
- All layer parameters must be fixed values inside the network, with no variables or external constants used.
- Avoid creating excessively large layers that would cause unreasonable memory usage.
- The model must only use standard layers directly, without defining helper functions or custom block classes.
- Output only the code, with no comments.
'''
import torch
import torch.nn as nn

class TextRNNBinaryClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings=800, embedding_dim=64)
        self.rnn1 = nn.RNN(input_size=64, hidden_size=64, num_layers=1, nonlinearity='tanh', batch_first=True, dropout=0.0, bidirectional=False)
        self.rnn2 = nn.RNN(input_size=64, hidden_size=64, num_layers=1, nonlinearity='tanh', batch_first=True, dropout=0.0, bidirectional=False)
        self.dropout = nn.Dropout(p=0.1)
        self.classifier = nn.Linear(in_features=64, out_features=1)

    def forward(self, x):
        x = self.embedding(x)
        out1, _ = self.rnn1(x)
        out1 = self.dropout(out1)
        out2, h2 = self.rnn2(out1)
        h = h2[-1]
        h = self.dropout(h)
        logits = self.classifier(h)
        return logits