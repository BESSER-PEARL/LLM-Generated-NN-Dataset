'''
Generate a complete PyTorch NN code that satisfies the following four Requirements: 

Architecture: CNN-1D 
Learning Task: Representation-learning
Input Type + Scale: Time series, seq_length: Large >2000 and features: Large >2000 
Complexity: Wide-Deep: Number of CNN-1D layers at least 4, out_channels of first Conv layer at least min(128, upper_bound_of_features).


- Provide complete PyTorch code for the model only.
- All layer parameters must be fixed values inside the network, with no variables or external constants used.
- Avoid creating excessively large layers that would cause unreasonable memory usage.
- The model must only use standard layers directly, without defining helper functions or custom block classes.
- Output only the code, with no comments.
'''
import torch
import torch.nn as nn

class WideDeepTS1DEncoder(nn.Module):
    def __init__(self):
        super(WideDeepTS1DEncoder, self).__init__()
        self.wide_conv = nn.Conv1d(in_channels=2048, out_channels=256, kernel_size=1, stride=1, padding=0, bias=False)
        self.wide_bn = nn.BatchNorm1d(256)
        self.wide_act = nn.ReLU(inplace=True)
        self.wide_pool = nn.AdaptiveAvgPool1d(1)

        self.deep_conv1 = nn.Conv1d(in_channels=2048, out_channels=128, kernel_size=7, stride=2, padding=3, bias=False)
        self.deep_bn1 = nn.BatchNorm1d(128)
        self.deep_act1 = nn.ReLU(inplace=True)
        self.deep_pool1 = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        self.deep_conv2 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=5, stride=2, padding=2, bias=False)
        self.deep_bn2 = nn.BatchNorm1d(256)
        self.deep_act2 = nn.ReLU(inplace=True)

        self.deep_conv3 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False)
        self.deep_bn3 = nn.BatchNorm1d(256)
        self.deep_act3 = nn.ReLU(inplace=True)
        self.deep_pool2 = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        self.deep_conv4 = nn.Conv1d(in_channels=256, out_channels=512, kernel_size=3, stride=2, padding=1, bias=False)
        self.deep_bn4 = nn.BatchNorm1d(512)
        self.deep_act4 = nn.ReLU(inplace=True)
        self.deep_pool3 = nn.AdaptiveAvgPool1d(1)

        self.proj_fc1 = nn.Linear(256 + 512, 512)
        self.proj_act = nn.ReLU(inplace=True)
        self.proj_drop = nn.Dropout(p=0.1)
        self.proj_fc2 = nn.Linear(512, 256)

    def forward(self, x):
        w = self.wide_conv(x)
        w = self.wide_bn(w)
        w = self.wide_act(w)
        w = self.wide_pool(w)
        w = torch.flatten(w, 1)

        d = self.deep_conv1(x)
        d = self.deep_bn1(d)
        d = self.deep_act1(d)
        d = self.deep_pool1(d)

        d = self.deep_conv2(d)
        d = self.deep_bn2(d)
        d = self.deep_act2(d)

        d = self.deep_conv3(d)
        d = self.deep_bn3(d)
        d = self.deep_act3(d)
        d = self.deep_pool2(d)

        d = self.deep_conv4(d)
        d = self.deep_bn4(d)
        d = self.deep_act4(d)
        d = self.deep_pool3(d)
        d = torch.flatten(d, 1)

        z = torch.cat([w, d], dim=1)
        z = self.proj_fc1(z)
        z = self.proj_act(z)
        z = self.proj_drop(z)
        z = self.proj_fc2(z)
        return z