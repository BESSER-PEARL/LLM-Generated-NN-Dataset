'''
Generate a complete PyTorch NN code that satisfies the following four Requirements: 

Architecture: RNN-Simple 
Learning Task: Classification-multiclass
Input Type + Scale: Text, seq_length: Small <50 and vocab_size: Small <1k 
Complexity: Simple: Number of RNN-Simple layers up to 2, embedding_dim of the Embedding layer up to min(128, upper_bound_of_vocab_size).


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
        self.embedding = nn.Embedding(num_embeddings=800, embedding_dim=64, padding_idx=0)
        self.rnn = nn.RNN(input_size=64, hidden_size=96, num_layers=2, nonlinearity='tanh', batch_first=True, dropout=0.0, bidirectional=False)
        self.dropout = nn.Dropout(p=0.1)
        self.fc = nn.Linear(in_features=96, out_features=15)

    def forward(self, x):
        x = self.embedding(x)
        out, hn = self.rnn(x)
        h = hn[-1]
        h = self.dropout(h)
        logits = self.fc(h)
        return logits