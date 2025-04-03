import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from torch.utils.data import DataLoader, TensorDataset
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.cuda.empty_cache()
print(f"Using device: {device}")

# File path to the dataset
file_path = 'new_data_urls.csv'
chunk_size = 10000

print("Loading and processing data in chunks...")
X_train_list = []
y_train_list = []

for chunk in pd.read_csv(file_path, chunksize=chunk_size):
    X_train_list.append(chunk['url'])
    y_train_list.append(chunk['status'])

X_train = pd.concat(X_train_list)
y_train = pd.concat(y_train_list)
del X_train_list, y_train_list

print("Tokenizing the text data...")
tokenizer = Tokenizer(char_level=True)
tokenizer.fit_on_texts(X_train)
X_train_sequences = tokenizer.texts_to_sequences(X_train)
max_sequence_length = min(300, max(len(seq) for seq in X_train_sequences))
X_train_padded = pad_sequences(X_train_sequences, maxlen=max_sequence_length, padding='post')

X_train_split, X_test_split, y_train_split, y_test_split = train_test_split(
    X_train_padded, y_train, test_size=0.2, random_state=42
)

# Convert labels to tensors (keep them on CPU for now)
y_train_split = torch.tensor(y_train_split.to_numpy(), dtype=torch.float32)
y_test_split = torch.tensor(y_test_split.to_numpy(), dtype=torch.float32)
X_train_split = torch.tensor(X_train_split, dtype=torch.long)
X_test_split = torch.tensor(X_test_split, dtype=torch.long)

# Create datasets (kept on CPU)
train_dataset = TensorDataset(X_train_split, y_train_split)
test_dataset = TensorDataset(X_test_split, y_test_split)

# Remove pin_memory=True
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Define LSTM Model
class LSTMPredictor(nn.Module):
    def __init__(self, vocab_size, embed_size=64, hidden_size=64, num_layers=2):
        super(LSTMPredictor, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_size, 1)  # No sigmoid here
    
    def forward(self, x):
        x = self.embedding(x)
        x, _ = self.lstm(x)
        x = self.fc(x[:, -1, :])  # No sigmoid here
        return x  # Raw logits output

model = LSTMPredictor(len(tokenizer.word_index) + 1).to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scaler = torch.cuda.amp.GradScaler()

print("Training the LSTM model on GPU...")

# Training loop
epochs = 10
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)  # Move to GPU inside loop
        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            outputs = model(inputs).squeeze()
            loss = criterion(outputs, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_loader)}")

# Evaluation
print("Evaluating the model...")
model.eval()
predictions, true_labels = [], []

import torch.nn.functional as F

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)  # Move inside loop
        outputs = model(inputs).squeeze()
        preds = (F.sigmoid(outputs) > 0.5).float()  # Apply sigmoid before thresholding
        predictions.extend(preds.cpu().numpy())
        true_labels.extend(labels.cpu().numpy())

accuracy = accuracy_score(true_labels, predictions)
print("Accuracy:", accuracy)
print("Classification Report:\n", classification_report(true_labels, predictions))

# Save the model and tokenizer
torch.save(model.state_dict(), 'lstm_url_phishing_model.pth')
joblib.dump(tokenizer, 'tokenizer.pkl')

print("Model and tokenizer saved.")
