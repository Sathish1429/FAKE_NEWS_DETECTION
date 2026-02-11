import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, f1_score
import os
from collections import Counter
import time
import torch.nn as nn
import re
import html

# === CONFIG ===
MODEL_NAME = 'bert-base-uncased'
BATCH_SIZE = 16
EPOCHS = 5
MAX_LEN = 256
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs('model_save', exist_ok=True)

# === CLEAN TEXT FUNCTION ===
def clean_text(text):
    text = str(text).lower()
    text = html.unescape(text)
    text = re.sub(r"http\S+|www\S+|https\S+", '', text)
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[\U00010000-\U0010ffff]', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# === DATA LOADING ===
real_df = pd.read_csv("real_news_4000.csv", encoding='utf-8')
fake_df = pd.read_csv("fake_news_4000.csv", encoding='utf-8')

if 'text' not in real_df.columns:
    real_df.rename(columns={real_df.columns[0]: 'text'}, inplace=True)
if 'text' not in fake_df.columns:
    fake_df.rename(columns={fake_df.columns[0]: 'text'}, inplace=True)

real_df['label'] = 0
fake_df['label'] = 1

# Equalize the number of real and fake samples
min_size = min(len(real_df), len(fake_df))
real_df = real_df.sample(n=min_size, random_state=42)
fake_df = fake_df.sample(n=min_size, random_state=42)

# Combine and shuffle
df = pd.concat([real_df, fake_df], ignore_index=True).sample(frac=1, random_state=42).reset_index(drop=True)

print("Label distribution after equalizing (0 = Real, 1 = Fake):")
print(df['label'].value_counts())



# === TOKENIZER & DATASET ===
tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)

class FakeNewsDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer.encode_plus(
            str(self.texts[idx]),
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': torch.tensor(self.labels[idx], dtype=torch.long)
        }
    

    df['text'] = df['text'].apply(clean_text)
    df = df[df['text'].str.split().str.len() > 5]
# Split
train_texts, val_texts, train_labels, val_labels = train_test_split(
    df['text'].tolist(),
    df['label'].tolist(),
    test_size=0.2,
    random_state=42,
    stratify=df['label'].tolist()
)

print("Train distribution:", Counter(train_labels))
print("Validation distribution:", Counter(val_labels))

train_dataset = FakeNewsDataset(train_texts, train_labels, tokenizer, MAX_LEN)
val_dataset = FakeNewsDataset(val_texts, val_labels, tokenizer, MAX_LEN)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

# === MODEL SETUP ===
model = BertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2).to(DEVICE)
optimizer = AdamW(model.parameters(), lr=2e-5)
loss_fn = nn.CrossEntropyLoss()

# === TRAINING LOOP ===
for epoch in range(EPOCHS):
    print(f"\nEpoch {epoch+1}/{EPOCHS}")
    model.train()
    total_loss = 0
    start = time.time()
    for batch in train_loader:
        input_ids = batch['input_ids'].to(DEVICE)
        attention_mask = batch['attention_mask'].to(DEVICE)
        labels = batch['labels'].to(DEVICE)

        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        loss = loss_fn(logits, labels)

        total_loss += loss.item()
        loss.backward()
        optimizer.step()

    print(f"Epoch Loss: {total_loss / len(train_loader):.4f}, Time: {time.time() - start:.2f}s")

# === EVALUATION ===
model.eval()
preds, true_labels = [], []

with torch.no_grad():
    for batch in val_loader:
        input_ids = batch['input_ids'].to(DEVICE)
        attention_mask = batch['attention_mask'].to(DEVICE)
        labels = batch['labels'].to(DEVICE)

        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=1)
        preds.extend(predictions.cpu().numpy())
        true_labels.extend(labels.cpu().numpy())

val_acc = accuracy_score(true_labels, preds)
val_f1 = f1_score(true_labels, preds, average='weighted')

print(f"\nValidation Accuracy: {val_acc:.4f}")
print(f"Validation F1 Score: {val_f1:.4f}")
print("\nClassification Report:")
print(classification_report(true_labels, preds, target_names=["Real", "Fake"]))

# === SAVE MODEL & TOKENIZER ===
model.save_pretrained("model_save/")
tokenizer.save_pretrained("model_save/")
print("\nModel and tokenizer saved to 'model_save/'")
