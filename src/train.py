import json
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from nltk_utils import tokenize, stem, bag_of_words
from model import AgriBotNN

# Seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

with open('src/data/intents.json', 'r', encoding='utf-8') as f:
    intents = json.load(f)

all_words = []
tags = []
xy = []

for intent in intents['intents']:
    tag = intent['tag']
    tags.append(tag)
    for pattern in intent['patterns']:
        w = tokenize(pattern)
        all_words.extend(w)
        xy.append((w, tag))

ignore_words = ['?', '!', '.', ',']
all_words = sorted(set([stem(w) for w in all_words if w not in ignore_words]))
tags = sorted(set(tags))

X_all = []
y_all = []
for (pattern_sentence, tag) in xy:
    bag = bag_of_words(pattern_sentence, all_words)
    X_all.append(bag)
    y_all.append(tags.index(tag))

# 80/20 Train-Validation Split to prevent overfitting
X_train, X_val, y_train, y_val = train_test_split(X_all, y_all, test_size=0.2, random_state=42)

X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
X_val = torch.tensor(X_val, dtype=torch.float32)
y_val = torch.tensor(y_val, dtype=torch.long)

class ChatDataset(Dataset):
    def __init__(self, x_data, y_data):
        self.n_samples = len(x_data)
        self.x_data = x_data
        self.y_data = y_data
    def __getitem__(self, index): return self.x_data[index], self.y_data[index]
    def __len__(self): return self.n_samples

train_dataset = ChatDataset(X_train, y_train)
train_loader = DataLoader(dataset=train_dataset, batch_size=8, shuffle=True)

# Hyperparameters
input_size = len(all_words)
hidden_size = 64 # Increased hidden size
output_size = len(tags)
learning_rate = 0.001
num_epochs = 500

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = AgriBotNN(input_size, hidden_size, output_size).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

print("Training AgriBot...")
for epoch in range(num_epochs):
    model.train()
    for words, labels in train_loader:
        words, labels = words.to(device), labels.to(device)
        outputs = model(words)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

# Validation Evaluation
model.eval()
with torch.no_grad():
    val_outputs = model(X_val.to(device))
    _, predicted = torch.max(val_outputs, dim=1)
    correct = (predicted == y_val.to(device)).sum().item()
    val_acc = correct / len(y_val)
    print(f"Final Validation Accuracy: {val_acc*100:.2f}%")

data = {
    "model_state": model.state_dict(),
    "input_size": input_size,
    "hidden_size": hidden_size,
    "output_size": output_size,
    "all_words": all_words,
    "tags": tags
}

torch.save(data, "src/data/agribot_data.pth")
print("Model saved to src/data/agribot_data.pth")