import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, get_linear_schedule_with_warmup
from torch.optim import AdamW
import torch
from tqdm import tqdm
from sklearn.metrics import classification_report
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from preprocessing import clean_text, lemmatize_text

# === КОНФІГУРАЦІЯ ===
MODEL_NAME = "bert-base-multilingual-cased"
NUM_LABELS = 6
BATCH_SIZE = 32
EPOCHS = 10
MAX_LEN = 256
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === КРОК 1. ЗАВАНТАЖЕННЯ ДАНИХ ===
df = pd.read_csv("merged_dataset.csv", encoding="utf-8-sig", sep=';')

# Чистимо та лематизуємо текст
df["clean_text"] = df["text"].apply(clean_text).apply(lemmatize_text)

# Перетворюємо label в integer
df["label"] = df["label"].astype(int)

# Train/test split
train_df, val_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df["label"])

# === КРОК 2. ДАТАСЕТ ===
class AggressionDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts.iloc[idx])
        label = int(self.labels.iloc[idx])
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            truncation=True,
            padding="max_length",
            return_attention_mask=True,
            return_tensors="pt"
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

# === КРОК 3. ТОКЕНІЗАТОР ===
tokenizer = BertTokenizer.from_pretrained(MODEL_NAME, timeout=60)

train_dataset = AggressionDataset(train_df["clean_text"], train_df["label"], tokenizer, MAX_LEN)
val_dataset = AggressionDataset(val_df["clean_text"], val_df["label"], tokenizer, MAX_LEN)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

# === КРОК 4. МОДЕЛЬ ===
model = BertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=NUM_LABELS)
model = model.to(DEVICE)

optimizer = AdamW(model.parameters(), lr=2e-5)
total_steps = len(train_loader) * EPOCHS
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

# === Балансування класів ===
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(train_df["label"]),
    y=train_df["label"]
)
class_weights = torch.tensor(class_weights, dtype=torch.float).to(DEVICE)
loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights)

# === КРОК 5. НАВЧАННЯ ===
for epoch in range(EPOCHS):
    print(f"\nEpoch {epoch + 1}/{EPOCHS}")
    model.train()
    train_loss = 0
    for batch in tqdm(train_loader, desc="Training"):
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(DEVICE)
        attention_mask = batch['attention_mask'].to(DEVICE)
        labels = batch['labels'].to(DEVICE)
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        train_loss += loss.item()
        loss.backward()
        optimizer.step()
        scheduler.step()
    avg_train_loss = train_loss / len(train_loader)
    print(f"Train Loss: {avg_train_loss:.4f}")

    # === ВАЛІДАЦІЯ ===
    model.eval()
    val_labels = []
    val_preds = []
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validation"):
            input_ids = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            labels = batch['labels'].to(DEVICE)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            preds = torch.argmax(logits, dim=1)
            val_labels.extend(labels.cpu().numpy())
            val_preds.extend(preds.cpu().numpy())
    report = classification_report(val_labels, val_preds, digits=3)
    print("\nValidation Report:\n", report)

# === ЗБЕРЕЖЕННЯ МОДЕЛІ ===
model.save_pretrained("saved_model")
tokenizer.save_pretrained("saved_model")
print("\nНавчання завершено. Модель збережено в папці 'saved_model'.")
