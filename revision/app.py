"""
URL Phishing Detection Application

This module provides functions to predict whether a URL is legitimate or phishing
using multiple models (XGBoost, LSTM, and DistilBERT) and ensemble methods.
"""

import numpy as np
import pandas as pd
import torch
import re
from urllib.parse import urlparse
import torch.nn.functional as F
from tqdm import tqdm
from scipy.sparse import hstack

def extract_url_features(url):
    """Extract domain-specific features from URLs"""
    features = {}
    
    # Basic URL structure
    features['length'] = len(url)
    
    try:
        parsed = urlparse(url)
        
        # URL components
        features['domain_length'] = len(parsed.netloc) if parsed.netloc else 0
        features['path_length'] = len(parsed.path) if parsed.path else 0
        features['query_length'] = len(parsed.query) if parsed.query else 0
        features['fragment_length'] = len(parsed.fragment) if parsed.fragment else 0
        
        # Domain specific
        domain = parsed.netloc
        features['subdomain_count'] = domain.count('.') if domain else 0
        
        # TLD length
        if domain and '.' in domain:
            features['tld_length'] = len(domain.split('.')[-1])
        else:
            features['tld_length'] = 0
            
    except:
        features['domain_length'] = 0
        features['path_length'] = 0
        features['query_length'] = 0
        features['fragment_length'] = 0
        features['subdomain_count'] = 0
        features['tld_length'] = 0
    
    # Character counts
    features['dots'] = url.count('.')
    features['hyphens'] = url.count('-')
    features['underscores'] = url.count('_')
    features['slashes'] = url.count('/')
    features['question_marks'] = url.count('?')
    features['equal_signs'] = url.count('=')
    features['at_symbols'] = url.count('@')
    features['ampersands'] = url.count('&')
    features['percent'] = url.count('%')
    
    # Character ratios
    features['digit_ratio'] = sum(c.isdigit() for c in url) / max(len(url), 1)
    features['uppercase_ratio'] = sum(c.isupper() for c in url) / max(len(url), 1)
    features['special_char_ratio'] = sum(not c.isalnum() for c in url) / max(len(url), 1)
    
    # Check for IP address
    features['has_ip_pattern'] = 1 if any(c.isdigit() and c != '.' for c in parsed.netloc.split('.')) else 0
    
    # Check for shortening services
    shortening_services = ['bit.ly', 'goo.gl', 't.co', 'tinyurl', 'is.gd', 'cli.gs', 'ow.ly', 'yfrog.com', 
                          'migre.me', 'ff.im', 'tiny.cc', 'url4.eu', 'twit.ac', 'su.pr', 'twurl.nl', 'snipurl.com']
    features['is_shortened'] = 1 if any(service in url.lower() for service in shortening_services) else 0
    
    # Check for suspicious words
    suspicious = ['login', 'secure', 'bank', 'account', 'update', 'verify', 'password', 'confirm', 
                 'pay', 'wallet', 'access', 'credit', 'bill', 'authenticate', 'ebay', 'paypal', 'uber']
    features['suspicious_words'] = sum(1 for word in suspicious if word in url.lower())
    
    # Security indicators
    features['has_https'] = 1 if url.lower().startswith('https') else 0
    
    return features

def predict_xgb(url, models):
    """
    Make a prediction using the XGBoost model.
    
    Args:
        url: URL to classify
        models: Dictionary of loaded models
        
    Returns:
        Prediction (1 for legitimate, 0 for phishing) or None if model not available
    """
    if models['xgb_bundle'] is None:
        return None
    
    try:
        # Extract features
        features = extract_url_features(url)
        
        # Convert to DataFrame
        df = pd.DataFrame([features])
        
        # Get components from the model bundle
        bundle = models['xgb_bundle']
        vectorizer = bundle.get('vectorizer')
        feature_selector = bundle.get('feature_selector')
        model = bundle.get('model') or bundle.get('ensemble')  # Try both keys
        
        if not model:
            return None
        
        # Apply TF-IDF vectorization if vectorizer exists
        if vectorizer:
            tfidf_features = vectorizer.transform([url])
            X = hstack([tfidf_features, df])
        else:
            X = df
            
        # Apply feature selection if available
        if feature_selector:
            X = feature_selector.transform(X)
            
        # Get probability
        if hasattr(model, 'predict_proba'):
            prob = model.predict_proba(X)[0, 1]  # Probability of legitimate class
        else:
            # If no predict_proba, use predict and convert to float
            prob = float(model.predict(X)[0])
            
        return prob
    except Exception as e:
        print(f"Error in XGBoost prediction: {e}")
        return None

def predict_lstm(url, models):
    """
    Make a prediction using the LSTM model.
    
    Args:
        url: URL to classify
        models: Dictionary of loaded models
        
    Returns:
        Prediction (1 for legitimate, 0 for phishing) or None if model not available
    """
    if models['lstm_model'] is None or models['lstm_tokenizer'] is None:
        return None
    
    try:
        # Tokenize the URL
        tokenizer = models['lstm_tokenizer']
        max_length = 100  # Adjust based on your model's input size
        
        # Convert URL to sequence
        sequence = tokenizer.texts_to_sequences([url])
        padded_seq = torch.nn.utils.rnn.pad_sequence(
            [torch.tensor(seq) for seq in sequence], 
            batch_first=True,
            padding_value=0
        )
        
        # Ensure sequence is within max_length
        if padded_seq.shape[1] > max_length:
            padded_seq = padded_seq[:, :max_length]
        
        # Make prediction
        with torch.no_grad():
            padded_seq = padded_seq.to(models['lstm_model'].embedding.weight.device)
            output = models['lstm_model'](padded_seq)
            prob = torch.sigmoid(output).item()
            
        return prob
    except Exception as e:
        print(f"Error in LSTM prediction: {e}")
        return None

def predict_bert(url, models):
    """
    Make a prediction using the DistilBERT model.
    
    Args:
        url: URL to classify
        models: Dictionary of loaded models
        
    Returns:
        Prediction (1 for legitimate, 0 for phishing) or None if model not available
    """
    if models['bert_model'] is None or models['bert_tokenizer'] is None:
        return None
    
    try:
        # Tokenize the URL
        tokenizer = models['bert_tokenizer']
        inputs = tokenizer(url, return_tensors="pt", truncation=True, padding=True)
        
        # Move inputs to the same device as model
        device = next(models['bert_model'].parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Make prediction
        with torch.no_grad():
            outputs = models['bert_model'](**inputs)
            logits = outputs.logits
            probabilities = F.softmax(logits, dim=1)
            prob = probabilities[0, 1].item()  # Probability of legitimate class
            
        return prob
    except Exception as e:
        print(f"Error in DistilBERT prediction: {e}")
        return None