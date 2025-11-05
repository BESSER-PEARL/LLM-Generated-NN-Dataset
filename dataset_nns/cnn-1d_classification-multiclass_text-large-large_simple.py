'''
Generate a complete PyTorch NN code that satisfies the following four Requirements: 

Architecture: CNN-1D 
Learning Task: Classification-multiclass
Input Type + Scale: Text, seq_length: Large >2000 and vocab_size: Large >100k 
Complexity: Simple: Number of CNN-1D layers up to 4, embedding_dim of the Embedding layer up to min(128, upper_bound_of_vocab_size).


- Provide complete PyTorch code for the model only.
- All layer parameters must be fixed values inside the network, with no variables or external constants used.
- Avoid creating excessively large layers that would cause unreasonable memory usage.
- The model must only use standard layers directly, without defining helper functions or custom block classes.
- Output only the code, with no comments.
'''
import torch
import torch.nn as nn

class TextCNN1D(nn.Module):
    def __init__(self):
        super(TextCNN1D, self).__init__()
        self.embedding = nn.Embedding(110000, 128, padding_idx=0)
        self.conv1 = nn.Conv1d(128, 128, kernel_size=5, padding=2)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool1d(kernel_size=4, stride=4)
        self.conv2 = nn.Conv1d(128, 128, kernel_size=5, padding=2)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool1d(kernel_size=4, stride=4)
        self.conv3 = nn.Conv1d(128, 128, kernel_size=3, padding=1)
        self.relu3 = nn.ReLU()
        self.global_pool = nn.AdaptiveMaxPool1d(1)
        self.flatten = nn.Flatten(1)
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(128, 10)

    def forward(self, x):
        x = self.embedding(x)
        x = x.transpose(1, 2)
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.relu3(x)
        x = self.global_pool(x)
        x = self.flatten(x)
        x = self.dropout(x)
        x = self.fc(x)
        return x