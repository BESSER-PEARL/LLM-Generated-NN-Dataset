'''
Generate a complete PyTorch NN code that satisfies the following four Requirements: 

Architecture: RNN-LSTM 
Learning Task: Classification-binary
Input Type + Scale: Text, seq_length: Large >2000 and vocab_size: Small <1k 
Complexity: Wide: embedding_dim of the Embedding layer at least min(128, upper_bound_of_vocab_size).


- Provide complete PyTorch code for the model only.
- All layer parameters must be fixed values inside the network, with no variables or external constants used.
- Avoid creating excessively large layers that would cause unreasonable memory usage.
- The model must only use standard layers directly, without defining helper functions or custom block classes.
- Output only the code, with no comments.
'''
import torch
import torch.nn as nn

class TextLSTMBinaryClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings=800, embedding_dim=256, padding_idx=0)
        self.lstm = nn.LSTM(input_size=256, hidden_size=256, num_layers=2, batch_first=True, bidirectional=True, dropout=0.2)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(512, 1)

    def forward(self, x):
        x = self.embedding(x)
        _, (h, _) = self.lstm(x)
        h_forward = h[-2]
        h_backward = h[-1]
        h_cat = torch.cat((h_forward, h_backward), dim=1)
        h_cat = self.dropout(h_cat)
        logits = self.fc(h_cat)
        return logits.squeeze(-1)