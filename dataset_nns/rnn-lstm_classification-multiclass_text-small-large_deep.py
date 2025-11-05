'''
Generate a complete PyTorch NN code that satisfies the following four Requirements: 

Architecture: RNN-LSTM 
Learning Task: Classification-multiclass
Input Type + Scale: Text, seq_length: Small <50 and vocab_size: Large >100k 
Complexity: Deep: Number of RNN-LSTM layers at least 2.


- Provide complete PyTorch code for the model only.
- All layer parameters must be fixed values inside the network, with no variables or external constants used.
- Avoid creating excessively large layers that would cause unreasonable memory usage.
- The model must only use standard layers directly, without defining helper functions or custom block classes.
- Output only the code, with no comments.
'''
import torch
import torch.nn as nn

class TextLSTMClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings=120000, embedding_dim=128, padding_idx=0)
        self.dropout_emb = nn.Dropout(p=0.2)
        self.lstm = nn.LSTM(input_size=128, hidden_size=128, num_layers=3, batch_first=True, dropout=0.2, bidirectional=False)
        self.dropout_fc = nn.Dropout(p=0.3)
        self.fc = nn.Linear(in_features=128, out_features=20)
    def forward(self, x):
        x = self.embedding(x)
        x = self.dropout_emb(x)
        output, (h_n, c_n) = self.lstm(x)
        h_last = h_n[-1]
        h_last = self.dropout_fc(h_last)
        logits = self.fc(h_last)
        return logits