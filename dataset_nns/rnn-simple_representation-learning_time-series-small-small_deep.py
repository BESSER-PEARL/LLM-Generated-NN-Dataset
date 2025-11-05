'''
Generate a complete PyTorch NN code that satisfies the following four Requirements: 

Architecture: RNN-Simple 
Learning Task: Representation-learning
Input Type + Scale: Time series, seq_length: Small <50 and features: Small <50 
Complexity: Deep: Number of RNN-Simple layers at least 2.


- Provide complete PyTorch code for the model only.
- All layer parameters must be fixed values inside the network, with no variables or external constants used.
- Avoid creating excessively large layers that would cause unreasonable memory usage.
- The model must only use standard layers directly, without defining helper functions or custom block classes.
- Output only the code, with no comments.
'''
import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc_rnn1 = nn.RNN(input_size=32, hidden_size=48, num_layers=1, batch_first=True, nonlinearity='tanh', bidirectional=False)
        self.enc_rnn2 = nn.RNN(input_size=48, hidden_size=48, num_layers=1, batch_first=True, nonlinearity='tanh', bidirectional=False)
        self.to_rep = nn.Linear(48, 24)
        self.from_rep_input = nn.Linear(24, 24)
        self.rep_to_h1 = nn.Linear(24, 48)
        self.rep_to_h2 = nn.Linear(24, 48)
        self.dec_rnn1 = nn.RNN(input_size=24, hidden_size=48, num_layers=1, batch_first=True, nonlinearity='tanh', bidirectional=False)
        self.dec_rnn2 = nn.RNN(input_size=48, hidden_size=48, num_layers=1, batch_first=True, nonlinearity='tanh', bidirectional=False)
        self.out_proj = nn.Linear(48, 32)

    def forward(self, x):
        y1, _ = self.enc_rnn1(x)
        y2, h2 = self.enc_rnn2(y1)
        rep = self.to_rep(h2[-1])
        b = x.size(0)
        t = x.size(1)
        dec_token = self.from_rep_input(rep)
        dec_in = dec_token.unsqueeze(1).repeat(1, t, 1)
        h0_1 = self.rep_to_h1(rep).unsqueeze(0)
        h0_2 = self.rep_to_h2(rep).unsqueeze(0)
        d1, _ = self.dec_rnn1(dec_in, h0_1)
        d2, _ = self.dec_rnn2(d1, h0_2)
        recon = self.out_proj(d2)
        return rep, recon