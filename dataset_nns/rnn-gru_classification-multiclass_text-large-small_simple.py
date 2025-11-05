'''
Generate a complete PyTorch NN code that satisfies the following four Requirements: 

Architecture: RNN-GRU 
Learning Task: Classification-multiclass
Input Type + Scale: Text, seq_length: Large >2000 and vocab_size: Small <1k 
Complexity: Simple: Number of RNN-GRU layers up to 2, embedding_dim of the Embedding layer up to min(128, upper_bound_of_vocab_size).


- Provide complete PyTorch code for the model only.
- All layer parameters must be fixed values inside the network, with no variables or external constants used.
- Avoid creating excessively large layers that would cause unreasonable memory usage.
- The model must only use standard layers directly, without defining helper functions or custom block classes.
- Output only the code, with no comments.
'''
import torch
import torch.nn as nn

class TextGRUClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings=800, embedding_dim=128, padding_idx=0)
        self.gru = nn.GRU(input_size=128, hidden_size=128, num_layers=2, batch_first=True, dropout=0.2)
        self.dropout = nn.Dropout(p=0.3)
        self.fc = nn.Linear(128, 10)

    def forward(self, x):
        x = self.embedding(x)
        output, h_n = self.gru(x)
        h_last = h_n[-1]
        h_last = self.dropout(h_last)
        logits = self.fc(h_last)
        return logits