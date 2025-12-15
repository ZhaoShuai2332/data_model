"""
Unified Data Preprocessing Module

This module handles data preprocessing for machine learning models.

Preprocessing steps:
1. Remove samples with missing values exceeding threshold
2. Remove columns with all zeros (no useful information)
3. Keep only numeric variables
4. Fill missing values with mean
5. Standardization
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


class DataPreprocessor:
    """Data preprocessing class for feature engineering and normalization."""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.feature_cols = None
        self.feature_means = None  # Store training set means for test set imputation
        
    def load_data(self, filepath):
        """Load data from CSV file."""
        print(f"Loading data: {filepath}")
        data = pd.read_csv(filepath)
        print(f"Data shape: {data.shape}")
        return data
    
    def preprocess_train(self, data, target_col='是否高风险', missing_threshold=0.5):
        """
        Preprocess training data.
        
        Args:
            data: Raw training data
            target_col: Target variable column name
            missing_threshold: Missing value ratio threshold for sample removal
            
        Returns:
            tuple: (X_scaled, y) preprocessed features and labels
        """
        print("\nStarting training data preprocessing...")
        print(f"Original data shape: {data.shape}")
        
        # 1. Calculate missing ratio per sample, remove samples with too many missing values
        missing_ratio = data.isnull().sum(axis=1) / data.shape[1]
        data = data[missing_ratio < missing_threshold].copy()
        print(f"After removing samples with >{int(missing_threshold*100)}% missing values: {data.shape}")
        
        # 2. Remove columns with all zeros
        # Exclude target and ID columns
        feature_cols = [col for col in data.columns if col not in [target_col, 'id']]
        
        # Select numeric features
        numeric_features = data[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
        
        # Remove all-zero columns
        zero_cols = [col for col in numeric_features if (data[col] == 0).all()]
        if zero_cols:
            data = data.drop(columns=zero_cols)
            print(f"Removed all-zero columns: {len(zero_cols)}")
        
        # Remove zero-variance columns
        var_zero_cols = [col for col in numeric_features 
                        if col in data.columns and data[col].var() == 0]
        if var_zero_cols:
            data = data.drop(columns=var_zero_cols)
            print(f"Removed zero-variance columns: {len(var_zero_cols)}")
        
        # 3. Update feature list (keep only numeric variables)
        self.feature_cols = [col for col in data.columns 
                            if col not in [target_col, 'id'] and 
                            col in data.select_dtypes(include=[np.number]).columns]
        
        print(f"Final number of features: {len(self.feature_cols)}")
        
        # 4. Fill missing values with mean
        self.feature_means = data[self.feature_cols].mean()
        data[self.feature_cols] = data[self.feature_cols].fillna(self.feature_means)
        
        # 5. Standardization
        X = data[self.feature_cols].values
        y = data[target_col].values
        
        X_scaled = self.scaler.fit_transform(X)
        
        print(f"Feature dimensions: {X_scaled.shape}, Label dimensions: {y.shape}")
        print(f"Positive sample ratio: {y.mean():.4f}")
        print(f"Negative sample ratio: {1 - y.mean():.4f}")
        
        return X_scaled, y
    
    def preprocess_test(self, data):
        """
        Preprocess test data.
        Uses training set feature list, means, and standardization parameters.
        
        Args:
            data: Raw test data
            
        Returns:
            ndarray: Preprocessed feature matrix
        """
        print("\nStarting test data preprocessing...")
        
        # Ensure test set has the same feature columns
        missing_cols = set(self.feature_cols) - set(data.columns)
        for col in missing_cols:
            data[col] = 0
        
        # Select the same feature columns
        data = data[self.feature_cols].copy()
        
        # Fill missing values with training set means
        data = data.fillna(self.feature_means)
        
        # Standardization (using training set parameters)
        X_scaled = self.scaler.transform(data.values)
        
        print(f"Test set feature dimensions: {X_scaled.shape}")
        
        return X_scaled
