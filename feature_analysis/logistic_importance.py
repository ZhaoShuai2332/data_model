# -*- coding: utf-8 -*-
"""
Logistic Regression Feature Importance Analysis

This script computes permutation importance for the trained Logistic Regression model
and saves the results to the outputs directory.

Usage:
    python -m feature_analysis.logistic_importance
    
    Or from project root:
    python feature_analysis/logistic_importance.py
"""

import os
import sys
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from feature_analysis.feature_importance_analysis import compute_importance_for_model


SEED = 42
np.random.seed(SEED)


def main():
    """Main function to compute feature importance for Logistic Regression."""
    
    print("=" * 60)
    print("Logistic Regression - Feature Importance Analysis")
    print("=" * 60)
    
    # Paths
    model_dir = os.path.join(project_root, 'logistic_model')
    output_dir = os.path.join(model_dir, 'outputs')
    
    # Load preprocessor to get feature names
    preprocessor_path = os.path.join(output_dir, 'preprocessor.pkl')
    if not os.path.exists(preprocessor_path):
        print(f"Error: Preprocessor not found at {preprocessor_path}")
        print("Please run the logistic model training first.")
        return
    
    preprocessor = joblib.load(preprocessor_path)
    feature_names = list(preprocessor.feature_cols)
    print(f"Loaded {len(feature_names)} feature names from preprocessor")
    
    # Load trained model
    model_path = os.path.join(output_dir, 'logistic_model_final.pkl')
    if not os.path.exists(model_path):
        model_path = os.path.join(output_dir, 'best_model.pkl')
    
    if not os.path.exists(model_path):
        print(f"Error: Model not found")
        print("Please run the logistic model training first.")
        return
    
    model = joblib.load(model_path)
    print(f"Loaded model from {model_path}")
    
    # Create a wrapper for sklearn model
    class ModelWrapper:
        def __init__(self, sklearn_model):
            self.model = sklearn_model
        
        def predict_proba(self, X):
            return self.model.predict_proba(X)
    
    model_wrapper = ModelWrapper(model)
    
    # Load and preprocess data
    train_data_path = os.path.join(project_root, 'train.csv')
    if not os.path.exists(train_data_path):
        print(f"Error: Training data not found at {train_data_path}")
        return
    
    print(f"\nLoading training data...")
    train_data = pd.read_csv(train_data_path)
    
    # Preprocess using saved preprocessor
    X, y = preprocessor.preprocess_train(train_data, target_col='是否高风险')
    
    # Split to get validation set (same as training)
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.3, random_state=SEED, stratify=y
    )
    
    print(f"Using validation set: {X_val.shape[0]} samples")
    
    # Compute permutation importance
    results = compute_importance_for_model(
        model=model_wrapper,
        X=X_val,
        y=y_val,
        feature_names=feature_names,
        output_dir=output_dir,
        model_name="Logistic Regression",
        n_repeats=5,
        random_state=SEED
    )
    
    print("\n" + "=" * 60)
    print("Feature importance analysis completed!")
    print("=" * 60)
    
    return results


if __name__ == '__main__':
    main()
