"""
Meta-Learner Training Script

This script trains a meta-learner model that finds optimal weights for combining
predictions from XGBoost, LSTM, and DistilBERT models for URL phishing detection.

Usage:
    python train_meta_learner.py --data_path <path_to_labeled_urls.csv> [--output_path <output_dir>]
"""
import argparse
import pandas as pd
import numpy as np
import os
import joblib
import json
import logging
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
import time
from tqdm import tqdm

# Import models and functions
from meta_learner import (
    LinearMetaLearner,
    NeuralMetaLearner,
    collect_base_model_predictions,
    train_meta_learner,
    load_models
)

# Set up logging
def setup_logger():
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    
    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    
    # Add handler to logger
    logger.addHandler(console_handler)
    
    return logger

logger = setup_logger()

def collect_predictions_with_progress(urls, labels, models, output_path=None):
    """
    Wrapper function to collect base model predictions with progress bar.
    """
    results = []
    
    with tqdm(total=len(urls), desc="Collecting predictions", unit="URL") as pbar:
        for i, url in enumerate(urls):
            row = {'url': url, 'true_label': labels[i]}
            
            # Get predictions from each model
            try:
                from app import predict_xgb, predict_lstm, predict_bert
                
                xgb_pred = predict_xgb(url, models)
                lstm_pred = predict_lstm(url, models)
                bert_pred = predict_bert(url, models)
                
                row['xgb_pred'] = xgb_pred if xgb_pred is not None else np.nan
                row['lstm_pred'] = lstm_pred if lstm_pred is not None else np.nan
                row['bert_pred'] = bert_pred if bert_pred is not None else np.nan
                
            except Exception as e:
                # Handle prediction errors gracefully
                row['xgb_pred'] = np.nan
                row['lstm_pred'] = np.nan
                row['bert_pred'] = np.nan
                logger.warning(f"Error predicting URL {i+1}: {e}")
            
            results.append(row)
            pbar.update(1)
    
    # Create DataFrame
    df = pd.DataFrame(results)
    
    # Save if output_path provided
    if output_path:
        df.to_csv(output_path, index=False)
    
    return df

def parse_args():
    parser = argparse.ArgumentParser(description='Train meta-learner for adaptive ensemble weights')
    parser.add_argument('--data_path', type=str, required=True, 
                        help='Path to CSV file with URLs and their labels (columns: url, label/status)')
    parser.add_argument('--output_path', type=str, default='./models/meta_learner',
                        help='Output directory to save meta-learner and results')
    parser.add_argument('--meta_learner_type', type=str, default='linear', choices=['linear', 'neural'],
                        help='Type of meta-learner to train')
    parser.add_argument('--cross_val', action='store_true', 
                        help='Perform cross-validation to tune meta-learner hyperparameters')
    parser.add_argument('--model_paths', type=str, default=None,
                        help='JSON file with custom paths for models')
    return parser.parse_args()

def main():
    # Parse command line arguments
    args = parse_args()
    
    # Create output directory if it doesn't exist
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)
        logger.info(f"Created output directory: {args.output_path}")
    
    # Load data
    logger.info(f"Loading data from {args.data_path}...")
    with tqdm(desc="Loading data", unit="rows") as pbar:
        data_df = pd.read_csv(args.data_path)
        pbar.update(len(data_df))
    logger.info(f"Loaded {len(data_df)} samples")
    
    # Check data format and standardize column names
    if 'url' not in data_df.columns:
        logger.error("Error: Data must contain a 'url' column")
        return
    
    # Check if we have a label column (either 'label' or 'status')
    if 'label' in data_df.columns:
        label_col = 'label'
    elif 'status' in data_df.columns:
        label_col = 'status'
    else:
        logger.error("Error: Data must contain either 'label' or 'status' column")
        return
    
    # Create a standardized true_label column for internal use
    data_df['true_label'] = data_df[label_col]
    
    # Load model paths if provided
    model_paths = None
    if args.model_paths:
        try:
            with open(args.model_paths, 'r') as f:
                model_paths = json.load(f)
            logger.info(f"Loaded custom model paths from {args.model_paths}")
        except Exception as e:
            logger.warning(f"Error loading model paths from {args.model_paths}: {e}")
            logger.warning("Using default model paths...")
    
    # Load base models
    logger.info("Loading base models...")
    with tqdm(desc="Loading models", total=3, unit="model") as pbar:
        models = load_models(model_paths)
        pbar.update(3)  # XGBoost, LSTM, BERT
    logger.info("Base models loaded successfully")
    
    # Collect predictions from base models
    logger.info("Collecting predictions from base models...")
    predictions_path = os.path.join(args.output_path, 'base_model_predictions.csv')
    
    # Using tqdm to display progress
    urls = data_df['url'].tolist()
    labels = data_df['true_label'].tolist()
    
    # Use our wrapper function with progress bar
    base_predictions_df = collect_predictions_with_progress(
        urls,
        labels,
        models,
        output_path=predictions_path
    )

    logger.info(f"Base model predictions collected and saved to {predictions_path}")
    
    # Print prediction statistics
    valid_preds = base_predictions_df.dropna()
    logger.info(f"Valid predictions: {len(valid_preds)} out of {len(base_predictions_df)} ({len(valid_preds)/len(base_predictions_df)*100:.2f}%)")
    
    # If performing cross-validation
    if args.cross_val and args.meta_learner_type == 'linear':
        logger.info("Performing hyperparameter tuning...")
        # Clean data - remove rows with missing predictions
        clean_df = base_predictions_df.dropna()
        
        # Extract features and target
        X = clean_df[['xgb_pred', 'lstm_pred', 'bert_pred']].values
        y = clean_df['true_label'].values
        
        # Setup grid search
        param_grid = {
            'C': [0.01, 0.1, 1.0, 10.0, 100.0],
            'class_weight': [None, 'balanced']
        }
        
        # Create base meta-learner
        base_meta = LinearMetaLearner().model
        
        # Perform grid search
        logger.info("Starting grid search for hyperparameter tuning...")
        with tqdm(desc="Grid Search CV", total=len(param_grid['C']) * len(param_grid['class_weight']) * 5, unit="fit") as pbar:
            grid_search = GridSearchCV(
                base_meta, param_grid, cv=5, scoring='accuracy', verbose=0
            )
            grid_search.fit(X, y)
            pbar.update(len(param_grid['C']) * len(param_grid['class_weight']) * 5)
        
        # Print best parameters
        logger.info(f"Best parameters: {grid_search.best_params_}")
        logger.info(f"Best cross-validation score: {grid_search.best_score_:.4f}")
        
        # Create meta-learner with best parameters
        if args.meta_learner_type == 'linear':
            meta_learner = LinearMetaLearner(**grid_search.best_params_)
        else:
            meta_learner = NeuralMetaLearner()
            
    else:
        # Train meta-learner without cross-validation
        logger.info(f"Training {args.meta_learner_type} meta-learner...")
        
        # Add progress indication for training
        with tqdm(desc=f"Training {args.meta_learner_type} meta-learner", total=1, unit="model") as pbar:
            meta_learner, weights = train_meta_learner(
                base_predictions_df,
                meta_learner_type=args.meta_learner_type,
                output_path=os.path.join(args.output_path, f'{args.meta_learner_type}_meta_learner.joblib')
            )
            pbar.update(1)
        
        # Save optimal weights
        weights_dict = {
            'xgboost': float(weights[0]),
            'lstm': float(weights[1]),
            'distilbert': float(weights[2]),
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        
        weights_path = os.path.join(args.output_path, 'optimal_weights.json')
        with tqdm(desc="Saving weights", total=1, unit="file") as pbar:
            with open(weights_path, 'w') as f:
                json.dump(weights_dict, f, indent=4)
            pbar.update(1)
        
        logger.info(f"Optimal weights saved to {weights_path}")
        logger.info(f"Weights: XGBoost={weights_dict['xgboost']:.3f}, LSTM={weights_dict['lstm']:.3f}, DistilBERT={weights_dict['distilbert']:.3f}")
        
        # Evaluate on a hold-out set
        logger.info("Evaluating meta-learner on a hold-out set...")
        # Clean data - remove rows with missing predictions
        clean_df = base_predictions_df.dropna()
        
        # Extract features and target
        X = clean_df[['xgb_pred', 'lstm_pred', 'bert_pred']].values
        y = clean_df['true_label'].values
        
        # Split into training and hold-out sets
        logger.info("Splitting data into training and hold-out sets...")
        with tqdm(desc="Splitting data", total=1, unit="split") as pbar:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            pbar.update(1)
        
        # Evaluate
        logger.info("Generating predictions on hold-out set...")
        with tqdm(desc="Evaluating meta-learner", total=len(X_test), unit="sample") as pbar:
            y_pred = meta_learner.predict(X_test)
            pbar.update(len(X_test))
        
        # Print evaluation metrics
        logger.info("\nGenerating classification report and confusion matrix...")
        with tqdm(desc="Generating reports", total=2, unit="report") as pbar:
            logger.info("\nClassification Report:")
            logger.info(classification_report(y_test, y_pred))
            pbar.update(1)
            
            # Print confusion matrix
            logger.info("\nConfusion Matrix:")
            logger.info(confusion_matrix(y_test, y_pred))
            pbar.update(1)
        
        # Compare with fixed weights
        logger.info("\nComparing with fixed weights (0.1, 0.3, 0.6):")
        fixed_weights = np.array([0.1, 0.3, 0.6])
        fixed_preds = []
        
        for i in tqdm(range(len(X_test)), desc="Evaluating fixed weights", unit="sample"):
            weighted_sum = np.dot(X_test[i], fixed_weights)
            fixed_preds.append(int(weighted_sum > 0.5))
            
        fixed_preds = np.array(fixed_preds)
        
        logger.info("\nGenerating fixed weights reports...")
        with tqdm(desc="Fixed weights reports", total=2, unit="report") as pbar:
            logger.info("Fixed Weights Classification Report:")
            logger.info(classification_report(y_test, fixed_preds))
            pbar.update(1)
            
            # Print confusion matrix
            logger.info("Fixed Weights Confusion Matrix:")
            logger.info(confusion_matrix(y_test, fixed_preds))
            pbar.update(1)
    
    logger.info("\nMeta-learner training complete!")
    logger.info(f"Model saved to {args.output_path}")

if __name__ == "__main__":
    main()