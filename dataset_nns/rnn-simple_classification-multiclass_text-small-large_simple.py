'''
Generate a complete PyTorch NN code that satisfies the following four Requirements: 

Architecture: RNN-Simple 
Learning Task: Classification-multiclass
Input Type + Scale: Text, seq_length: Small <50 and vocab_size: Large >100k 
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
        self.embedding = nn.Embedding(num_embeddings=120000, embedding_dim=96, padding_idx=0)
        self.rnn = nn.RNN(input_size=96, hidden_size=128, num_layers=1, batch_first=True, nonlinearity='tanh', bidirectional=False)
        self.fc = nn.Linear(in_features=128, out_features=10)

    def forward(self, x):
        x = self.embedding(x)
        output, hidden = self.rnn(x)
        logits = self.fc(hidden[-1])
        return logits