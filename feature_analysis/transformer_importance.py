# -*- coding: utf-8 -*-
"""
Transformer Feature Importance Analysis

This script computes permutation importance for the trained Transformer model
and saves the results to the outputs directory.

Usage:
    python -m feature_analysis.transformer_importance
    
    Or from project root:
    python feature_analysis/transformer_importance.py
"""

import os
import sys
import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Add transformer_model to path for imports
transformer_model_dir = os.path.join(project_root, 'transformer_model')
if transformer_model_dir not in sys.path:
    sys.path.insert(0, transformer_model_dir)

from feature_analysis.feature_importance_analysis import compute_importance_for_model


SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class TransformerModelWrapper:
    """Wrapper for Transformer model to provide predict interface."""
    
    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.model.eval()
    
    def predict(self, X):
        """Predict probabilities for input X."""
        self.model.eval()
        X_tensor = torch.FloatTensor(X).to(self.device)
        
        with torch.no_grad():
            outputs, _ = self.model(X_tensor)  # Transformer returns (output, attention_weights)
            predictions = outputs.cpu().numpy()
        
        return predictions


def main():
    """Main function to compute feature importance for Transformer."""
    
    print("=" * 60)
    print("Transformer - Feature Importance Analysis")
    print("=" * 60)
    print(f"Using device: {device}")
    
    # Paths
    model_dir = os.path.join(project_root, 'transformer_model')
    output_dir = os.path.join(model_dir, 'outputs')
    
    # Load preprocessor to get feature names
    preprocessor_path = os.path.join(output_dir, 'preprocessor.pkl')
    if not os.path.exists(preprocessor_path):
        print(f"Error: Preprocessor not found at {preprocessor_path}")
        print("Please run the Transformer model training first.")
        return
    
    preprocessor = joblib.load(preprocessor_path)
    feature_names = list(preprocessor.feature_cols)
    print(f"Loaded {len(feature_names)} feature names from preprocessor")
    
    # Load model checkpoint
    model_path = os.path.join(output_dir, 'transformer_model_final.pth')
    if not os.path.exists(model_path):
        model_path = os.path.join(output_dir, 'best_model.pth')
    
    if not os.path.exists(model_path):
        print(f"Error: Model not found")
        print("Please run the Transformer model training first.")
        return
    
    checkpoint = torch.load(model_path, map_location=device)
    input_dim = checkpoint['input_dim']
    
    print(f"Loaded checkpoint from {model_path}")
    print(f"Input dimension: {input_dim}")
    
    # Import and create Transformer model
    from transformer import TabularTransformer
    
    model = TabularTransformer(
        input_dim=input_dim,
        d_model=128,
        num_heads=8,
        num_layers=4,
        d_ff=512,
        dropout=0.1
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Create wrapper
    model_wrapper = TransformerModelWrapper(model, device)
    
    # Load and preprocess data
    train_data_path = os.path.join(project_root, 'train.csv')
    if not os.path.exists(train_data_path):
        print(f"Error: Training data not found at {train_data_path}")
        return
    
    print(f"\nLoading training data...")
    train_data = pd.read_csv(train_data_path)
    
    # Preprocess using saved preprocessor
    X, y = preprocessor.preprocess_train(train_data, target_col='是否高风险')
    
    # Split to get validation set
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y
    )
    
    print(f"Using validation set: {X_val.shape[0]} samples")
    
    # Compute permutation importance
    results = compute_importance_for_model(
        model=model_wrapper,
        X=X_val,
        y=y_val,
        feature_names=feature_names,
        output_dir=output_dir,
        model_name="Transformer",
        n_repeats=5,
        random_state=SEED
    )
    
    print("\n" + "=" * 60)
    print("Feature importance analysis completed!")
    print("=" * 60)
    
    return results


if __name__ == '__main__':
    main()
