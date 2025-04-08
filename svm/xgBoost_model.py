import os
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
from xgboost import XGBClassifier

# -----------------------------------------
# 1. Load Dataset and Basic EDA
# -----------------------------------------

# Update the file path as needed
file_path = "/kaggle/input/url-detection-dataset/corrected_preprocessed_urls.csv"
df = pd.read_csv(file_path)

# Display basic information about the dataset
print("Dataset shape:", df.shape)
print(df.head())

# Distribution of the target variable
plt.figure(figsize=(6,4))
sns.countplot(data=df, x="status")
plt.title("Distribution of URL Status")
plt.xlabel("Status (0: Legit, 1: Phishing)")
plt.ylabel("Count")
plt.show()

# Plot URL length distribution (optional, since URLs can vary a lot)
df["url_length"] = df["url"].apply(len)
plt.figure(figsize=(8,4))
sns.histplot(df["url_length"], bins=50, kde=True)
plt.title("Distribution of URL Lengths")
plt.xlabel("URL Length")
plt.ylabel("Frequency")
plt.show()

# -----------------------------------------
# 2. Feature Extraction with TF-IDF
# -----------------------------------------
vectorizer = TfidfVectorizer(analyzer='char_wb', ngram_range=(3,5), max_features=5000)
X = vectorizer.fit_transform(df["url"])
y = df["status"]

# -----------------------------------------
# 3. Train-Test Split
# -----------------------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
print("Training set:", X_train.shape, "Test set:", X_test.shape)

# -----------------------------------------
# 4. Train XGBoost Model
# -----------------------------------------
xgb_model = XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.1, use_label_encoder=False, eval_metric='logloss', random_state=42)
xgb_model.fit(X_train, y_train)

# -----------------------------------------
# 5. Evaluation
# -----------------------------------------

# Predictions
y_pred = xgb_model.predict(X_test)
y_pred_prob = xgb_model.predict_proba(X_test)[:, 1]

# Accuracy and Classification Report
accuracy = accuracy_score(y_test, y_pred)
print("Test Accuracy: {:.4f}".format(accuracy))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Confusion Matrix Visualization
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap="Blues", cbar=False)
plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()

# ROC Curve Visualization
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
roc_auc = auc(fpr, tpr)
plt.figure(figsize=(7,5))
plt.plot(fpr, tpr, label="ROC curve (area = {:.4f})".format(roc_auc))
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Receiver Operating Characteristic")
plt.legend(loc="lower right")
plt.show()

# -----------------------------------------
# 6. Save the Model as a PKL File
# -----------------------------------------
model_filename = "xgboost_url_model.pkl"
with open(model_filename, "wb") as f:
    pickle.dump((xgb_model, vectorizer), f)
print("\nXGBoost model and vectorizer saved as '{}'".format(model_filename))
