import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc, precision_recall_curve
from tqdm import tqdm

# 🚀 Check for CUDA and Free VRAM
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.cuda.empty_cache()
print(f"Using device: {device}")

# Load Preprocessed Dataset
file_path = "/kaggle/input/newwwww/corrected_preprocessed_urls.csv"  # Update if needed
df = pd.read_csv(file_path)

# Split dataset into training and testing
X_train, X_test, y_train, y_test = train_test_split(df["url"], df["status"], test_size=0.2, random_state=42)

# Tokenizer
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

# Custom Dataset Class
class URLDataset(Dataset):
    def __init__(self, urls, labels, tokenizer, max_len=128):
        self.urls = urls
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.encodings = tokenizer(list(urls), truncation=True, padding="max_length", max_length=self.max_len)

    def __len__(self):
        return len(self.urls)

    def __getitem__(self, idx):
        return {
            "input_ids": torch.tensor(self.encodings["input_ids"][idx], dtype=torch.long),
            "attention_mask": torch.tensor(self.encodings["attention_mask"][idx], dtype=torch.long),
            "labels": torch.tensor(self.labels.iloc[idx], dtype=torch.long),
        }

# Create PyTorch DataLoaders
train_dataset = URLDataset(X_train, y_train, tokenizer)
test_dataset = URLDataset(X_test, y_test, tokenizer)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0, pin_memory=True)

# Initialize Model
model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2).to(device)

# Freeze Lower Layers of DistilBERT
for param in model.distilbert.transformer.layer[:3].parameters():
    param.requires_grad = False

# Define Loss and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=2e-5)
scaler = torch.cuda.amp.GradScaler()

# Training Loop
epochs = 3
gradient_accumulation_steps = 4
loss_history = []

print("🚀 Training DistilBERT Model...")

for epoch in range(epochs):
    model.train()
    total_loss = 0
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")

    optimizer.zero_grad()

    for i, batch in enumerate(progress_bar):
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)

        with torch.cuda.amp.autocast():
            # Forward Pass
            outputs = model(input_ids, attention_mask=attention_mask).logits
            loss = criterion(outputs, labels.long()) 

        scaler.scale(loss).backward()

        if (i + 1) % gradient_accumulation_steps == 0 or (i + 1) == len(train_loader):
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        total_loss += loss.item()
        progress_bar.set_postfix(loss=loss.item())

        if device.type == "cuda":
            torch.cuda.empty_cache()

    avg_loss = total_loss / len(train_loader)
    loss_history.append(avg_loss)
    print(f"✅ Epoch {epoch+1}/{epochs} | Avg Loss: {avg_loss:.4f}")

# Evaluation
print("🔎 Evaluating Model...")
model.eval()
predictions, true_labels, prediction_probs = [], [], []

with torch.no_grad():
    for batch in tqdm(test_loader, desc="Evaluating"):
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)

        outputs = model(input_ids, attention_mask=attention_mask).logits  
        probs = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()  # Probabilities of class 1 (phishing)
        preds = torch.argmax(outputs, dim=1).cpu().numpy()

        predictions.extend(preds)
        true_labels.extend(labels.cpu().numpy())
        prediction_probs.extend(probs)

accuracy = accuracy_score(true_labels, predictions)
print(f"🎯 Accuracy: {accuracy:.4f}")
print("📊 Classification Report:\n", classification_report(true_labels, predictions))

# Save Model
model.save_pretrained("bert_url_model")
tokenizer.save_pretrained("bert_tokenizer")
print("✅ Model and tokenizer saved successfully!")

### 🔥 Visualizations ###

# 1️⃣ Confusion Matrix
conf_matrix = confusion_matrix(true_labels, predictions)
plt.figure(figsize=(6, 5))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=["Non-Phishing", "Phishing"], yticklabels=["Non-Phishing", "Phishing"])
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()
