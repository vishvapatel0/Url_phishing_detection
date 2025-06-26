"""
Meta-Learner for Adaptive Ensemble Weight Learning

This module implements a meta-learner that learns optimal weights for combining 
predictions from multiple base models (XGBoost, LSTM, DistilBERT) for URL phishing detection.
The meta-learner uses model stacking to adaptively learn weights based on validation data,
addressing the research critique about arbitrary weight assignment.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, cross_val_score
import joblib
import pickle
import torch
import os

from urllib.parse import urlparse
from scipy.sparse import hstack
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Import prediction functions for URL phishing detection
from app import predict_xgb, predict_lstm, predict_bert

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



def load_models(model_paths=None):
    """
    Load all models and return them in a dictionary
    
    Args:
        model_paths: Dictionary with custom paths for models. If None, uses default paths.
    """
    if model_paths is None:
        model_paths = {
            "xgb": "url_model_bundle.pkl",
            "lstm_tokenizer": "lstm/tokenizer.pkl",
            "lstm_model": "lstm/lstm_url_phishing_model.pth",
            "bert_tokenizer": "distilBert/bert_tokenizer",
            "bert_model": "distilBert/bert_url_model"
        }
    
    models = {}
    
    # Load XGBoost model bundle
    try:
        with open(model_paths["xgb"], "rb") as f:
            bundle = pickle.load(f)
            # Make sure the bundle has the right structure
            if isinstance(bundle, dict):
                models["xgb_bundle"] = bundle
            else:
                # If not a dict, create a dict with the model
                models["xgb_bundle"] = {"model": bundle}
    except Exception as e:
        print(f"Error loading XGBoost model: {e}")
        models["xgb_bundle"] = None
    
    # Rest of the function remains the same...
    
    # Load LSTM tokenizer and model
    try:
        models["lstm_tokenizer"] = joblib.load(model_paths["lstm_tokenizer"])
        
        # LSTM Model Definition
        class LSTMPredictor(nn.Module):
            def __init__(self, vocab_size, embed_size=64, hidden_size=64, num_layers=2):
                super(LSTMPredictor, self).__init__()
                self.embedding = nn.Embedding(vocab_size, embed_size)
                self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
                self.fc = nn.Linear(hidden_size, 1)
            
            def forward(self, x):
                x = self.embedding(x)
                x, _ = self.lstm(x)
                x = self.fc(x[:, -1, :])
                return x
        
        models["lstm_model"] = LSTMPredictor(len(models["lstm_tokenizer"].word_index) + 1).to(device)
        models["lstm_model"].load_state_dict(torch.load(model_paths["lstm_model"], map_location=device))
        models["lstm_model"].eval()
    except Exception as e:
        print(f"Error loading LSTM model: {e}")
        models["lstm_tokenizer"] = None
        models["lstm_model"] = None
    
    # Load DistilBERT tokenizer and model
    try:
        models["bert_tokenizer"] = AutoTokenizer.from_pretrained(model_paths["bert_tokenizer"])
        models["bert_model"] = AutoModelForSequenceClassification.from_pretrained(model_paths["bert_model"])
        models["bert_model"].to(device)
        models["bert_model"].eval()
    except Exception as e:
        print(f"Error loading DistilBERT model: {e}")
        models["bert_tokenizer"] = None
        models["bert_model"] = None
    
    return models

# Define meta-learner classes
class LinearMetaLearner:
    """
    A meta-learner that uses logistic regression to learn optimal ensemble weights.
    """
    def __init__(self, C=1.0, class_weight='balanced'):
        """
        Initialize the LinearMetaLearner with logistic regression.
        
        Args:
            C: Inverse of regularization strength
            class_weight: Weights associated with classes for imbalanced datasets
        """
        self.model = LogisticRegression(C=C, class_weight=class_weight, solver='liblinear')
        self.weights = None
        
    def fit(self, base_predictions, true_labels):
        """
        Train the meta-learner on base model predictions.
        
        Args:
            base_predictions: Array of shape (n_samples, n_models) with predictions from base models
            true_labels: Array of shape (n_samples,) with ground truth labels
        """
        self.model.fit(base_predictions, true_labels)
        # Extract learned weights - coefficients from logistic regression
        self.weights = self.model.coef_[0]
        # Normalize weights to sum to 1 and ensure non-negative values
        self.weights = np.maximum(0, self.weights)
        if np.sum(self.weights) > 0:
            self.weights = self.weights / np.sum(self.weights)
        else:
            # Fallback to uniform weights if all weights are non-positive
            self.weights = np.ones(len(self.weights)) / len(self.weights)
        return self.weights
    
    def predict(self, base_predictions):
        """
        Make predictions using the meta-learner.
        
        Args:
            base_predictions: Array of shape (n_samples, n_models) with predictions from base models
            
        Returns:
            Array of predictions
        """
        return self.model.predict(base_predictions)
    
    def predict_proba(self, base_predictions):
        """
        Make probability predictions using the meta-learner.
        
        Args:
            base_predictions: Array of shape (n_samples, n_models) with predictions from base models
            
        Returns:
            Array of probability predictions
        """
        return self.model.predict_proba(base_predictions)
    
    def get_weights(self):
        """
        Get the learned weights for the ensemble.
        
        Returns:
            Array of weights for each base model
        """
        return self.weights
    
    def save(self, path):
        """
        Save the meta-learner to disk.
        
        Args:
            path: Path to save the model
        """
        joblib.dump(self, path)
    
    @classmethod
    def load(cls, path):
        """
        Load the meta-learner from disk.
        
        Args:
            path: Path to load the model from
            
        Returns:
            Loaded meta-learner
        """
        return joblib.load(path)


class NeuralMetaLearner:
    """
    A meta-learner that uses a neural network to learn optimal ensemble weights.
    More flexible than linear meta-learner but requires more data.
    """
    def __init__(self, hidden_layer_sizes=(10, 5), activation='relu', alpha=0.0001):
        """
        Initialize the NeuralMetaLearner with an MLP classifier.
        
        Args:
            hidden_layer_sizes: Size of hidden layers
            activation: Activation function
            alpha: L2 regularization parameter
        """
        self.model = MLPClassifier(
            hidden_layer_sizes=hidden_layer_sizes, 
            activation=activation,
            alpha=alpha,
            max_iter=1000,
            random_state=42
        )
        self.weights = None
    
    def fit(self, base_predictions, true_labels):
        """
        Train the meta-learner on base model predictions.
        
        Args:
            base_predictions: Array of shape (n_samples, n_models) with predictions from base models
            true_labels: Array of shape (n_samples,) with ground truth labels
        """
        self.model.fit(base_predictions, true_labels)
        
        # For neural networks, we need to estimate weights by evaluating importance
        # Using permutation importance as a proxy for weights
        weights = []
        baseline_score = self.model.score(base_predictions, true_labels)
        
        for i in range(base_predictions.shape[1]):
            # Create a copy and permute one feature
            X_permuted = base_predictions.copy()
            X_permuted[:, i] = np.random.permutation(X_permuted[:, i])
            
            # Score after permutation
            permuted_score = self.model.score(X_permuted, true_labels)
            
            # Importance is the decrease in performance
            importance = max(0, baseline_score - permuted_score)
            weights.append(importance)
        
        self.weights = np.array(weights)
        
        # Normalize weights to sum to 1
        if np.sum(self.weights) > 0:
            self.weights = self.weights / np.sum(self.weights)
        else:
            # Fallback to uniform weights
            self.weights = np.ones(len(self.weights)) / len(self.weights)
            
        return self.weights
    
    def predict(self, base_predictions):
        """
        Make predictions using the meta-learner.
        
        Args:
            base_predictions: Array of shape (n_samples, n_models) with predictions from base models
            
        Returns:
            Array of predictions
        """
        return self.model.predict(base_predictions)
    
    def predict_proba(self, base_predictions):
        """
        Make probability predictions using the meta-learner.
        
        Args:
            base_predictions: Array of shape (n_samples, n_models) with predictions from base models
            
        Returns:
            Array of probability predictions
        """
        return self.model.predict_proba(base_predictions)
    
    def get_weights(self):
        """
        Get the learned weights for the ensemble.
        
        Returns:
            Array of weights for each base model
        """
        return self.weights
    
    def save(self, path):
        """
        Save the meta-learner to disk.
        
        Args:
            path: Path to save the model
        """
        joblib.dump(self, path)
    
    @classmethod
    def load(cls, path):
        """
        Load the meta-learner from disk.
        
        Args:
            path: Path to load the model from
            
        Returns:
            Loaded meta-learner
        """
        return joblib.load(path)


# Helper functions for meta-learning
def collect_base_model_predictions(urls, labels, models, output_path=None):
    """
    Collect predictions from all base models for meta-learner training.
    
    Args:
        urls: List of URLs
        labels: True labels for URLs (1 for legitimate, 0 for phishing)
        models: Dictionary of loaded models
        output_path: Path to save the predictions
        
    Returns:
        DataFrame with base model predictions and true labels
    """
    results = []
    
    for i, url in enumerate(urls):
        row = {'url': url, 'true_label': labels[i]}
        
        # Get predictions from each model
        xgb_pred = predict_xgb(url, models)
        lstm_pred = predict_lstm(url, models)
        bert_pred = predict_bert(url, models)
        
        row['xgb_pred'] = xgb_pred if xgb_pred is not None else np.nan
        row['lstm_pred'] = lstm_pred if lstm_pred is not None else np.nan
        row['bert_pred'] = bert_pred if bert_pred is not None else np.nan
        
        results.append(row)
    
    # Create DataFrame
    df = pd.DataFrame(results)
    
    # Save if output_path provided
    if output_path:
        df.to_csv(output_path, index=False)
    
    return df


def train_meta_learner(predictions_df, meta_learner_type='linear', output_path=None):
    """
    Train a meta-learner on base model predictions.
    
    Args:
        predictions_df: DataFrame with base model predictions and true labels
        meta_learner_type: Type of meta-learner ('linear' or 'neural')
        output_path: Path to save the trained meta-learner
        
    Returns:
        Trained meta-learner and optimal weights
    """
    # Clean data - remove rows with missing predictions
    clean_df = predictions_df.dropna()
    
    # Extract features and target
    X = clean_df[['xgb_pred', 'lstm_pred', 'bert_pred']].values
    y = clean_df['true_label'].values
    
    # Split into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Initialize and train meta-learner
    if meta_learner_type.lower() == 'neural':
        meta_learner = NeuralMetaLearner()
    else:
        meta_learner = LinearMetaLearner()
    
    # Fit and get weights
    weights = meta_learner.fit(X_train, y_train)
    
    # Evaluate performance
    val_score = meta_learner.model.score(X_val, y_val)
    print(f"Meta-learner validation accuracy: {val_score:.4f}")
    print(f"Optimal weights: {weights}")
    
    # Save meta-learner if output path provided
    if output_path:
        meta_learner.save(output_path)
    
    return meta_learner, weights


def ensemble_adaptive_vote(url, models, meta_learner=None):
    """
    Make predictions using the adaptive ensemble approach.
    
    Args:
        url: URL to classify
        models: Dictionary of loaded models
        meta_learner: Trained meta-learner
        
    Returns:
        Final prediction (1 for legitimate, 0 for phishing) and confidence
    """
    # Get individual predictions
    xgb_pred = predict_xgb(url, models)
    lstm_pred = predict_lstm(url, models)
    bert_pred = predict_bert(url, models)
    
    # If we don't have a meta-learner, use uniform weights
    if meta_learner is None:
        weights = np.array([1/3, 1/3, 1/3])
    else:
        weights = meta_learner.get_weights()
    
    # Check if any model failed to make a prediction
    valid_preds = []
    valid_weights = []
    
    if xgb_pred is not None:
        valid_preds.append(xgb_pred)
        valid_weights.append(weights[0])
        
    if lstm_pred is not None:
        valid_preds.append(lstm_pred)
        valid_weights.append(weights[1])
        
    if bert_pred is not None:
        valid_preds.append(bert_pred)
        valid_weights.append(weights[2])
    
    # If no valid predictions, return None
    if not valid_preds:
        return None, 0
    
    # Normalize weights
    valid_weights = np.array(valid_weights)
    valid_weights = valid_weights / np.sum(valid_weights)
    
    # Compute weighted average
    weighted_sum = np.dot(valid_preds, valid_weights)
    
    # Final prediction and confidence
    final_pred = int(weighted_sum > 0.5)
    confidence = abs(weighted_sum - 0.5) * 2  # Scale to [0, 1]
    
    return final_pred, confidence


if __name__ == "__main__":
    # Example of training a meta-learner
    # This would typically be done in a separate training script
    print("This module provides tools for meta-learning adaptive ensemble weights.")
    print("Import and use these functions in your main application.")