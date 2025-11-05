'''
Generate a complete PyTorch NN code that satisfies the following four Requirements: 

Architecture: RNN-Simple 
Learning Task: Classification-binary
Input Type + Scale: Text, seq_length: Small <50 and vocab_size: Small <1k 
Complexity: Wide-Deep: Number of RNN-Simple layers at least 2, embedding_dim of the Embedding layer at least min(128, upper_bound_of_vocab_size).


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
        super(TextRNNBinaryClassifier, self).__init__()
        self.embedding = nn.Embedding(num_embeddings=800, embedding_dim=128, padding_idx=0)
        self.rnn = nn.RNN(input_size=128, hidden_size=64, num_layers=3, nonlinearity='tanh', batch_first=True, bidirectional=True, dropout=0.2)
        self.dropout = nn.Dropout(p=0.3)
        self.fc1 = nn.Linear(384, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        x = self.embedding(x)
        _, h_n = self.rnn(x)
        rnn_feat = torch.cat((h_n[-2], h_n[-1]), dim=1)
        mean_pool = x.mean(dim=1)
        max_pool = x.max(dim=1).values
        wide_feat = torch.cat((mean_pool, max_pool), dim=1)
        combined = torch.cat((rnn_feat, wide_feat), dim=1)
        combined = self.dropout(combined)
        out = self.fc1(combined)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        return out