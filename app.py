import streamlit as st
import pickle
import joblib
import numpy as np
from urllib.parse import urlparse
from scipy.sparse import hstack
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd
import time
import os

# Set page configuration
st.set_page_config(
    page_title="URL Phishing Detector",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        text-align: center;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #424242;
        text-align: center;
        margin-bottom: 2rem;
    }
    .result-safe {
        font-size: 2rem;
        color: #4CAF50;
        text-align: center;
        padding: 1rem;
        border-radius: 10px;
        border: 2px solid #4CAF50;
        background-color: rgba(76, 175, 80, 0.1);
    }
    .result-phishing {
        font-size: 2rem;
        color: #F44336;
        text-align: center;
        padding: 1rem;
        border-radius: 10px;
        border: 2px solid #F44336;
        background-color: rgba(244, 67, 54, 0.1);
    }
    .model-details {
        font-size: 1rem;
        color: #424242;
        margin-top: 2rem;
    }
    .footer {
        text-align: center;
        margin-top: 3rem;
        color: #9E9E9E;
        font-size: 0.8rem;
    }
</style>
""", unsafe_allow_html=True)

# ----------------------#
# DEVICE CONFIGURATION
# ----------------------#
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def pad_sequences_custom(sequences, maxlen, padding='post', value=0):
    padded = np.full((len(sequences), maxlen), fill_value=value)
    for idx, seq in enumerate(sequences):
        if not seq:
            continue
        if padding == 'post':
            trunc = seq[:maxlen]
            padded[idx, :len(trunc)] = trunc
        elif padding == 'pre':
            trunc = seq[-maxlen:]
            padded[idx, -len(trunc):] = trunc
    return padded

# ----------------------#
# FEATURE EXTRACTION
# ----------------------#
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

# ----------------------#
# LOAD MODELS
# ----------------------#
@st.cache_resource
def load_models():
    """Load all models and return them in a dictionary"""
    models = {}
    
    # Load XGBoost model bundle
    try:
        st.info("Loading XGBoost model...")
        with open("url_model_bundle.pkl", "rb") as f:
            models["xgb_bundle"] = pickle.load(f)
        st.success("XGBoost model loaded successfully!")
    except Exception as e:
        st.error(f"Error loading XGBoost model: {e}")
        models["xgb_bundle"] = None
    
    # Load LSTM tokenizer and model
    try:
        st.info("Loading LSTM model...")
        models["lstm_tokenizer"] = joblib.load("lstm/tokenizer.pkl")
        
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
        models["lstm_model"].load_state_dict(torch.load("lstm/lstm_url_phishing_model.pth", map_location=device))
        models["lstm_model"].eval()
        st.success("LSTM model loaded successfully!")
    except Exception as e:
        st.error(f"Error loading LSTM model: {e}")
        models["lstm_tokenizer"] = None
        models["lstm_model"] = None
    
    # Load DistilBERT tokenizer and model
    try:
        st.info("Loading DistilBERT model...")
        models["bert_tokenizer"] = AutoTokenizer.from_pretrained("distilBert/bert_tokenizer")
        models["bert_model"] = AutoModelForSequenceClassification.from_pretrained("distilBert/bert_url_model")
        models["bert_model"].to(device)
        models["bert_model"].eval()
        st.success("DistilBERT model loaded successfully!")
    except Exception as e:
        st.error(f"Error loading DistilBERT model: {e}")
        models["bert_tokenizer"] = None
        models["bert_model"] = None
    
    return models

# ----------------------#
# PREDICTION FUNCTIONS
# ----------------------#
def predict_xgb(url, models):
    """Predict using XGBoost model"""
    if models["xgb_bundle"] is None:
        return None
    
    try:
        loaded_model = models["xgb_bundle"]["model"]
        loaded_vectorizer = models["xgb_bundle"]["vectorizer"]
        loaded_selector = models["xgb_bundle"]["feature_selector"]

        # Extract features for the URL
        url_features = extract_url_features(url)
        custom_features_df = pd.DataFrame([url_features])
        tfidf_vector = loaded_vectorizer.transform([url])
        
        # Combine features
        combined = hstack([tfidf_vector, custom_features_df])
        
        # Apply feature selection
        selected = loaded_selector.transform(combined)
        
        # Predict
        raw_prediction = loaded_model.predict(selected)[0]
        # Invert if needed to ensure 1=legitimate, 0=phishing
        prediction = raw_prediction  # Adjust this based on your model's original output
        return prediction
    except Exception as e:
        st.warning(f"XGBoost prediction error: {e}")
        return None

def predict_lstm(url, models):
    """Predict using LSTM model"""
    if models["lstm_model"] is None or models["lstm_tokenizer"] is None:
        return None
    
    try:
        sequence = models["lstm_tokenizer"].texts_to_sequences([url])
        padded = pad_sequences_custom(sequence, maxlen=300, padding='post')
        input_tensor = torch.tensor(padded, dtype=torch.long).to(device)

        with torch.no_grad():
            output = models["lstm_model"](input_tensor).squeeze()
            probability = torch.sigmoid(output).item()
            # Invert if needed to ensure 1=legitimate, 0=phishing
            prediction = 1 if probability > 0.5 else 0
        return prediction
    except Exception as e:
        st.warning(f"LSTM prediction error: {e}")
        return None

def predict_bert(url, models):
    """Predict using DistilBERT model"""
    if models["bert_model"] is None or models["bert_tokenizer"] is None:
        return None
    
    try:
        inputs = models["bert_tokenizer"](url, return_tensors="pt", truncation=True, padding="max_length", max_length=128)
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)

        with torch.no_grad():
            logits = models["bert_model"](input_ids=input_ids, attention_mask=attention_mask).logits
            probs = F.softmax(logits, dim=1)
            raw_prediction = torch.argmax(logits, dim=1).item()
            # Invert if needed to ensure 1=legitimate, 0=phishing
            prediction = raw_prediction  # Adjust this based on your model's original output

        return prediction
    except Exception as e:
        st.warning(f"DistilBERT prediction error: {e}")
        return None

def ensemble_majority_vote(url, models):
    """Ensemble prediction using majority voting"""
    predictions = []
    
    xgb_pred = predict_xgb(url, models)
    if xgb_pred is not None:
        predictions.append(xgb_pred)
    
    lstm_pred = predict_lstm(url, models)
    if lstm_pred is not None:
        predictions.append(lstm_pred)
    
    bert_pred = predict_bert(url, models)
    if bert_pred is not None:
        predictions.append(bert_pred)
    
    if not predictions:
        return None
    
    # Count votes for legitimate (1)
    legitimate_votes = sum(predictions)
    # Majority voting - if more than half are legitimate, return 1 (legitimate)
    return int(legitimate_votes > len(predictions)/2)

def ensemble_weighted_vote(url, models, weights):
    """Ensemble prediction using weighted voting"""
    total_weight = 0
    weighted_sum = 0
    
    xgb_pred = predict_xgb(url, models)
    if xgb_pred is not None:
        weighted_sum += xgb_pred * weights[0]
        total_weight += weights[0]
    
    lstm_pred = predict_lstm(url, models)
    if lstm_pred is not None:
        weighted_sum += lstm_pred * weights[1]
        total_weight += weights[1]
    
    bert_pred = predict_bert(url, models)
    if bert_pred is not None:
        weighted_sum += bert_pred * weights[2]
        total_weight += weights[2]
    
    if total_weight == 0:
        return None
    
    normalized_sum = weighted_sum / total_weight
    # If weighted sum > 0.5, return 1 (legitimate)
    return int(normalized_sum > 0.5)

# ----------------------#
# STREAMLIT APP LAYOUT
# ----------------------#
def main():
    # Header
    st.markdown('<h1 class="main-header">URL Phishing Detection</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Check if a URL is safe or potentially malicious</p>', unsafe_allow_html=True)
    
    # Load models
    with st.spinner("Loading models... This may take a moment."):
        models = load_models()
    
    # Create tabs
    tab1, tab2, tab3 = st.tabs(["URL Check", "Batch Processing", "About"])
    
    # Tab 1: Single URL Check
    with tab1:
        st.subheader("Enter a URL to check")
        url = st.text_input("URL:", placeholder="https://example.com")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            prediction_method = st.radio(
                "Prediction Method",
                ["Ensemble (Weighted)", "Ensemble (Majority)", "XGBoost", "LSTM", "DistilBERT"],
                index=0
            )
        
        with col2:
            # Fix for the slider issue - using individual sliders instead of a tuple slider
            if prediction_method == "Ensemble (Weighted)":
                st.write("Model Weights:")
                xgb_weight = st.slider("XGBoost Weight", 0.0, 1.0, 0.1, 0.1)
                lstm_weight = st.slider("LSTM Weight", 0.0, 1.0, 0.3, 0.1)
                bert_weight = st.slider("DistilBERT Weight", 0.0, 1.0, 0.6, 0.1)
                # Create weights tuple
                weights = (xgb_weight, lstm_weight, bert_weight)
            else:
                weights = (0.1, 0.3, 0.6)  # Default weights
        
        with col3:
            st.write("Device:", device)
            st.write("Models loaded:", sum([1 for k, v in models.items() if v is not None and k not in ["lstm_tokenizer", "bert_tokenizer"]]))
        
        if st.button("Check URL"):
            if not url:
                st.warning("Please enter a URL.")
            else:
                with st.spinner("Analyzing URL..."):
                    # Prediction timer
                    start_time = time.time()
                    
                    # Get prediction based on selected method
                    if prediction_method == "XGBoost":
                        prediction = predict_xgb(url, models)
                    elif prediction_method == "LSTM":
                        prediction = predict_lstm(url, models)
                    elif prediction_method == "DistilBERT":
                        prediction = predict_bert(url, models)
                    elif prediction_method == "Ensemble (Majority)":
                        prediction = ensemble_majority_vote(url, models)
                    else:  # Ensemble (Weighted)
                        prediction = ensemble_weighted_vote(url, models, weights)
                    
                    end_time = time.time()
                    
                    if prediction is None:
                        st.error("Error during prediction. Please check if all models are loaded correctly.")
                    else:
                        # Display Results
                        st.subheader("Analysis Results")
                        
                        # 1 = legitimate, 0 = phishing
                        if prediction == 1:
                            st.markdown('<div class="result-safe">✅ LEGITIMATE URL</div>', unsafe_allow_html=True)
                        else:
                            st.markdown('<div class="result-phishing">⚠️ POTENTIAL PHISHING DETECTED</div>', unsafe_allow_html=True)
                        
                        st.write(f"Prediction time: {end_time - start_time:.4f} seconds")
                        
                        # Display individual model predictions
                        st.subheader("Individual Model Predictions")
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            xgb_pred = predict_xgb(url, models)
                            st.metric("XGBoost", "Legitimate" if xgb_pred == 1 else "Phishing" if xgb_pred is not None else "Error")
                        
                        with col2:
                            lstm_pred = predict_lstm(url, models)
                            st.metric("LSTM", "Legitimate" if lstm_pred == 1 else "Phishing" if lstm_pred is not None else "Error")
                        
                        with col3:
                            bert_pred = predict_bert(url, models)
                            st.metric("DistilBERT", "Legitimate" if bert_pred == 1 else "Phishing" if bert_pred is not None else "Error")
                        
                        # Feature Analysis
                        st.subheader("URL Feature Analysis")
                        features = extract_url_features(url)
                        
                        # Display parsed URL components
                        parsed = urlparse(url)
                        st.write("URL Components:")
                        components = {
                            "Scheme": parsed.scheme,
                            "Netloc": parsed.netloc,
                            "Path": parsed.path,
                            "Params": parsed.params,
                            "Query": parsed.query,
                            "Fragment": parsed.fragment
                        }
                        st.json(components)
                        
                        # Display important features
                        st.write("Key Features:")
                        col1, col2 = st.columns(2)
                        
                        important_features = {
                            "URL Length": features["length"],
                            "Domain Length": features["domain_length"],
                            "Path Length": features["path_length"],
                            "HTTPS": "Yes" if features["has_https"] == 1 else "No",
                            "Suspicious Words": features["suspicious_words"],
                            "Is Shortened": "Yes" if features["is_shortened"] == 1 else "No",
                            "Special Character Ratio": f"{features['special_char_ratio']:.2f}",
                            "Digit Ratio": f"{features['digit_ratio']:.2f}"
                        }
                        
                        with col1:
                            for k, v in list(important_features.items())[:4]:
                                st.metric(k, v)
                        
                        with col2:
                            for k, v in list(important_features.items())[4:]:
                                st.metric(k, v)
    
    # Tab 2: Batch Processing
    with tab2:
        st.subheader("Batch URL Processing")
        
        st.write("Upload a CSV file with URLs to check")
        st.write("The file should have a column named 'url' containing the URLs to check.")
        
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
        
        if uploaded_file is not None:
            # Read the CSV file
            try:
                df = pd.read_csv(uploaded_file)
                
                if 'url' not in df.columns:
                    st.error("CSV file must contain a column named 'url'")
                else:
                    st.write(f"Found {len(df)} URLs to process")
                    
                    # Display sample
                    st.write("Sample URLs:")
                    st.dataframe(df.head())
                    
                    # Prediction method
                    batch_prediction_method = st.radio(
                        "Batch Prediction Method",
                        ["Ensemble (Weighted)", "Ensemble (Majority)", "XGBoost", "LSTM", "DistilBERT"],
                        index=0,
                        key="batch_prediction_method"
                    )
                    
                    if batch_prediction_method == "Ensemble (Weighted)":
                        st.write("Batch Model Weights:")
                        batch_xgb_weight = st.slider("XGBoost Weight", 0.0, 1.0, 0.1, 0.1, key="batch_xgb_weight")
                        batch_lstm_weight = st.slider("LSTM Weight", 0.0, 1.0, 0.3, 0.1, key="batch_lstm_weight")
                        batch_bert_weight = st.slider("DistilBERT Weight", 0.0, 1.0, 0.6, 0.1, key="batch_bert_weight")
                        batch_weights = (batch_xgb_weight, batch_lstm_weight, batch_bert_weight)
                    else:
                        batch_weights = (0.1, 0.3, 0.6)  # Default weights
                    
                    if st.button("Process Batch"):
                        progress_bar = st.progress(0)
                        results = []
                        
                        for i, row in enumerate(df.itertuples()):
                            url = getattr(row, 'url')
                            
                            # Get prediction based on selected method
                            if batch_prediction_method == "XGBoost":
                                prediction = predict_xgb(url, models)
                            elif batch_prediction_method == "LSTM":
                                prediction = predict_lstm(url, models)
                            elif batch_prediction_method == "DistilBERT":
                                prediction = predict_bert(url, models)
                            elif batch_prediction_method == "Ensemble (Majority)":
                                prediction = ensemble_majority_vote(url, models)
                            else:  # Ensemble (Weighted)
                                prediction = ensemble_weighted_vote(url, models, batch_weights)
                            
                            results.append({
                                'url': url,
                                'prediction': 'Legitimate' if prediction == 1 else 'Phishing' if prediction is not None else 'Error'
                            })
                            
                            # Update progress
                            progress_bar.progress((i + 1) / len(df))
                        
                        # Create results dataframe
                        results_df = pd.DataFrame(results)
                        
                        # Display results
                        st.subheader("Batch Processing Results")
                        st.dataframe(results_df)
                        
                        # Summary statistics
                        legitimate_count = sum(1 for r in results if r['prediction'] == 'Legitimate')
                        phishing_count = sum(1 for r in results if r['prediction'] == 'Phishing')
                        error_count = sum(1 for r in results if r['prediction'] == 'Error')
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Legitimate URLs", legitimate_count)
                        with col2:
                            st.metric("Phishing URLs", phishing_count)
                        with col3:
                            st.metric("Errors", error_count)
                        
                        # Download results button
                        csv = results_df.to_csv(index=False)
                        st.download_button(
                            label="Download Results as CSV",
                            data=csv,
                            file_name="phishing_detection_results.csv",
                            mime="text/csv",
                        )
                        
            except Exception as e:
                st.error(f"Error processing file: {e}")
    
    # Tab 3: About
    with tab3:
        st.subheader("About this Phishing Detection System")
        
        st.write("""
        This application uses an ensemble of machine learning models to detect phishing URLs:
        
        1. **XGBoost Model**: A gradient boosting model that uses engineered URL features and TF-IDF vectorization.
        
        2. **LSTM (Long Short-Term Memory) Model**: A deep learning model that processes URLs as sequences of characters.
        
        3. **DistilBERT Model**: A transformer-based model that captures complex patterns in URL text.
        
        The ensemble combines these models using either majority voting or weighted voting for improved accuracy.
        """)
        
        st.subheader("How it Works")
        
        st.write("""
        1. **URL Feature Extraction**: 
           The system extracts various features from the URL such as length, domain characteristics, character distributions, and presence of suspicious patterns.
        
        2. **Model Predictions**: 
           Each model independently analyzes the URL and makes a prediction.
        
        3. **Ensemble Decision**: 
           The final prediction is determined by combining the individual model predictions based on the selected ensemble method.
        
        The system classifies URLs as either legitimate (1) or phishing (0).
        """)
        
        st.subheader("Usage Tips")
        
        st.write("""
        - For quick checks of individual URLs, use the "URL Check" tab.
        - For processing multiple URLs, use the "Batch Processing" tab and upload a CSV file.
        - Try different ensemble methods to see how they affect the predictions.
        - Look at the individual model predictions to understand how each model classifies a URL.
        """)
        
        st.write("""
        **Note**: While this system is designed to detect phishing URLs, it may not catch all malicious URLs. 
        Always exercise caution when clicking on unfamiliar links.
        """)
    
    # Footer
    st.markdown('<div class="footer">URL Phishing Detection System | Powered by Machine Learning & Deep Learning</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
