"""
Evaluate NN on TEXT input data using a benchmark dataset (AG News).
"""

from collections import Counter
import torch
from torch import nn, optim
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset




dataset = load_dataset("ag_news")
tokenizer = lambda x: x.lower().split()
all_texts = [tokenizer(x) for x in dataset['train']['text']]

# Build vocab
counter = Counter(token for text in all_texts for token in text)
common = counter.most_common(50000)
vocab = {word: idx+1 for idx, (word, _) in enumerate(common)}
vocab["<pad>"] = 0

MAX_LEN = 200

def encode(text):
    tokens = tokenizer(text)
    ids = [vocab.get(t, 0) for t in tokens][:MAX_LEN]
    return torch.tensor(ids, dtype=torch.long)

class AGNewsDataset(Dataset):
    def __init__(self, split):
        self.data = dataset[split]
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        text = encode(self.data[idx]['text'])
        label = self.data[idx]['label']
        return text, label

def collate_batch(batch):
    texts, labels = zip(*batch)
    texts = pad_sequence(texts, batch_first=True, padding_value=0)
    labels = torch.tensor(labels, dtype=torch.long)
    return texts, labels


train_loader = DataLoader(AGNewsDataset('train'), batch_size=64,
                          shuffle=True, collate_fn=collate_batch)
test_loader = DataLoader(AGNewsDataset('test'), batch_size=64,
                         shuffle=False, collate_fn=collate_batch)




# Adapted from dataset_nns/cnn-1d_classification-multiclass_text-large-large_simple.py
class TextCNN1D(nn.Module):
    def __init__(self):
        super(TextCNN1D, self).__init__()
        self.embedding = nn.Embedding(len(vocab), 128, padding_idx=0)
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
        self.fc = nn.Linear(128, 20)

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


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = TextCNN1D().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

EPOCHS = 10
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        outputs = model(xb)
        loss = criterion(outputs, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * xb.size(0)
    total_loss /= len(train_loader.dataset)
    print(f"Epoch {epoch+1}/{EPOCHS} - Train Loss: {total_loss:.4f}")


# Evaluation
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for xb, yb in test_loader:
        xb, yb = xb.to(device), yb.to(device)
        outputs = model(xb)
        preds = outputs.argmax(dim=1)
        correct += (preds == yb).sum().item()
        total += yb.size(0)

accuracy = correct / total
print(f"Test Accuracy: {accuracy:.4f}")
