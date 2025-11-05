'''
Generate a complete PyTorch NN code that satisfies the following four Requirements: 

Architecture: RNN-Simple 
Learning Task: Representation-learning
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

class TextRNNRepresentation(nn.Module):
    def __init__(self):
        super(TextRNNRepresentation, self).__init__()
        self.embedding = nn.Embedding(num_embeddings=131072, embedding_dim=128, padding_idx=0)
        self.rnn1 = nn.RNN(input_size=128, hidden_size=256, num_layers=1, nonlinearity='tanh', batch_first=True, bidirectional=True)
        self.rnn2 = nn.RNN(input_size=512, hidden_size=256, num_layers=1, nonlinearity='tanh', batch_first=True, bidirectional=True)
        self.pool = nn.AdaptiveAvgPool1d(output_size=1)
        self.norm = nn.LayerNorm(1024)
        self.fc1 = nn.Linear(1024, 512)
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(512, 256)

    def forward(self, x):
        x = self.embedding(x)
        o1, _ = self.rnn1(x)
        o2, _ = self.rnn2(o1)
        last = o2[:, -1, :]
        mean = self.pool(o2.transpose(1, 2)).squeeze(-1)
        h = torch.cat([last, mean], dim=1)
        h = self.norm(h)
        h = self.fc1(h)
        h = self.act(h)
        h = self.fc2(h)
        return h