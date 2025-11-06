"""
Evaluate NN on Time Series input data using a benchmark dataset (HAR data).
"""
import zipfile
import io
import urllib.request
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler




url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00240/UCI%20HAR%20Dataset.zip"
with urllib.request.urlopen(url) as resp:
    with zipfile.ZipFile(io.BytesIO(resp.read())) as zf:
        zf.extractall("har_data")


class HARDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def load_har(path="har_data/UCI HAR Dataset"):
    x_tr = pd.read_csv(f"{path}/train/X_train.txt",
                          delim_whitespace=True, header=None)
    y_tr = pd.read_csv(f"{path}/train/y_train.txt",
                          delim_whitespace=True, header=None)
    x_ts = pd.read_csv(f"{path}/test/X_test.txt",
                         delim_whitespace=True, header=None)
    y_ts = pd.read_csv(f"{path}/test/y_test.txt",
                         delim_whitespace=True, header=None)

    scaler = StandardScaler()
    x_tr = scaler.fit_transform(x_tr)
    x_ts = scaler.transform(x_ts)

    x_tr = np.expand_dims(x_tr, 1)
    x_ts = np.expand_dims(x_ts, 1)

    return x_tr, y_tr.values.flatten()-1, x_ts, y_ts.values.flatten()-1

X_train, y_train, X_test, y_test = load_har()

train_ds = HARDataset(X_train, y_train)
test_ds = HARDataset(X_test, y_test)
train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=64)



# Adapted from dataset_nns/rnn-gru_classification-multiclass_time-series-large-small_wide.py
class RNNGRUClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.gru1 = nn.GRU(input_size=561, hidden_size=128, num_layers=1, batch_first=True, bidirectional=True)
        self.dropout1 = nn.Dropout(p=0.3)
        self.gru2 = nn.GRU(input_size=256, hidden_size=96, num_layers=1, batch_first=True, bidirectional=True)
        self.dropout2 = nn.Dropout(p=0.3)
        self.fc1 = nn.Linear(in_features=192, out_features=64)
        self.relu = nn.ReLU()
        self.fc_out = nn.Linear(in_features=64, out_features=6)

    def forward(self, x):
        out1, _ = self.gru1(x)
        out1 = self.dropout1(out1)
        _, h2 = self.gru2(out1)
        h_forward = h2[0]
        h_backward = h2[1]
        h_cat = torch.cat((h_forward, h_backward), dim=1)
        h_cat = self.dropout2(h_cat)
        z = self.fc1(h_cat)
        z = self.relu(z)
        logits = self.fc_out(z)
        return logits


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = RNNGRUClassifier().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)


EPOCHS = 10
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for X, y in train_loader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(X)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * X.size(0)
    tr_loss = total_loss / len(train_loader.dataset)
    print(f"Epoch {epoch+1}/{EPOCHS} - Train Loss: {tr_loss:.4f}")


model.eval()
correct, total = 0, 0
with torch.no_grad():
    for X, y in test_loader:
        X, y = X.to(device), y.to(device)
        preds = model(X).argmax(1)
        correct += (preds == y).sum().item()
        total += y.size(0)
print(f"Test Accuracy: {correct/total:.4f}")
