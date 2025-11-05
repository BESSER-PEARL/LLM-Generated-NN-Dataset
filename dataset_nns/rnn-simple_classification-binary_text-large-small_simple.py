'''
Generate a complete PyTorch NN code that satisfies the following four Requirements: 

Architecture: RNN-Simple 
Learning Task: Classification-binary
Input Type + Scale: Text, seq_length: Large >2000 and vocab_size: Small <1k 
Complexity: Simple: Number of RNN-Simple layers up to 2, embedding_dim of the Embedding layer up to min(128, upper_bound_of_vocab_size).


- Provide complete PyTorch code for the model only.
- All layer parameters must be fixed values inside the network, with no variables or external constants used.
- Avoid creating excessively large layers that would cause unreasonable memory usage.
- The model must only use standard layers directly, without defining helper functions or custom block classes.
- Output only the code, with no comments.
'''
import torch
import torch.nn as nn

class RNNBinaryClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings=800, embedding_dim=128, padding_idx=0)
        self.rnn = nn.RNN(input_size=128, hidden_size=64, num_layers=2, nonlinearity='tanh', batch_first=True, dropout=0.1, bidirectional=False)
        self.dropout = nn.Dropout(p=0.2)
        self.fc = nn.Linear(64, 1)

    def forward(self, x):
        x = self.embedding(x)
        x = self.dropout(x)
        output, hn = self.rnn(x)
        h_last = hn[-1]
        h_last = self.dropout(h_last)
        logits = self.fc(h_last)
        return logits