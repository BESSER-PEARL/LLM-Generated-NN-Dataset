'''
Generate a complete PyTorch NN code that satisfies the following four Requirements: 

Architecture: RNN-Simple 
Learning Task: Classification-binary
Input Type + Scale: Text, seq_length: Small <50 and vocab_size: Large >100k 
Complexity: Wide: embedding_dim of the Embedding layer at least min(128, upper_bound_of_vocab_size).


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
        self.embedding = nn.Embedding(num_embeddings=120000, embedding_dim=128, padding_idx=0)
        self.rnn = nn.RNN(input_size=128, hidden_size=64, num_layers=1, batch_first=True, nonlinearity='tanh', bidirectional=False)
        self.fc = nn.Linear(64, 1)

    def forward(self, input_ids):
        x = self.embedding(input_ids)
        _, h_n = self.rnn(x)
        x = h_n[-1]
        x = self.fc(x)
        return x.squeeze(-1)