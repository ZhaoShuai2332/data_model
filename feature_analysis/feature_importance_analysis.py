# -*- coding: utf-8 -*-
"""
Feature Importance Analysis Module

This module provides tools for computing permutation importance for any model.
Permutation importance measures the decrease in model performance when a 
feature's values are randomly shuffled.

References:
    - Breiman, L. (2001). Random Forests. Machine Learning, 45(1), 5-32.
    - Fisher, A., Rudin, C., & Dominici, F. (2019). All Models are Wrong, 
      but Many are Useful. JMLR.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from typing import Callable, List, Optional, Union
import warnings
warnings.filterwarnings('ignore')


class PermutationImportance:
    """
    Compute permutation importance for feature analysis.
    
    Permutation importance is calculated by measuring the decrease in 
    model performance when a single feature's values are randomly shuffled.
    
    Attributes:
        model: A trained model with predict or predict_proba method
        feature_names: List of feature names
        n_repeats: Number of times to permute each feature
        random_state: Random seed for reproducibility
    """
    
    def __init__(self, 
                 model,
                 feature_names: List[str],
                 n_repeats: int = 5,
                 random_state: int = 42):
        """
        Initialize PermutationImportance.
        
        Args:
            model: Trained model with predict/predict_proba method
            feature_names: List of feature names
            n_repeats: Number of permutation repeats (default: 5)
            random_state: Random seed (default: 42)
        """
        self.model = model
        self.feature_names = feature_names
        self.n_repeats = n_repeats
        self.random_state = random_state
        self.importances_ = None
        self.importances_mean_ = None
        self.importances_std_ = None
        
    def _get_predictions(self, X: np.ndarray) -> np.ndarray:
        """
        Get model predictions (probability for classification).
        
        Args:
            X: Feature matrix
            
        Returns:
            Prediction probabilities or values
        """
        if hasattr(self.model, 'predict_proba'):
            # For sklearn-style models
            return self.model.predict_proba(X)[:, 1]
        elif hasattr(self.model, 'predict'):
            # For custom models (MLP, Transformer)
            preds = self.model.predict(X)
            if isinstance(preds, np.ndarray):
                return preds.flatten()
            return preds
        else:
            raise ValueError("Model must have predict or predict_proba method")
    
    def _calculate_score(self, 
                         y_true: np.ndarray, 
                         y_pred: np.ndarray) -> float:
        """
        Calculate model performance score (ROC-AUC).
        
        Args:
            y_true: True labels
            y_pred: Predicted probabilities
            
        Returns:
            ROC-AUC score
        """
        return roc_auc_score(y_true, y_pred)
    
    def compute(self, 
                X: np.ndarray, 
                y: np.ndarray,
                scoring: Optional[Callable] = None) -> pd.DataFrame:
        """
        Compute permutation importance for all features.
        
        Args:
            X: Feature matrix (n_samples, n_features)
            y: Target labels
            scoring: Custom scoring function (default: ROC-AUC)
            
        Returns:
            DataFrame with feature, importance_mean, importance_std
        """
        np.random.seed(self.random_state)
        
        if scoring is None:
            scoring = self._calculate_score
        
        n_features = X.shape[1]
        
        # Calculate baseline score
        baseline_pred = self._get_predictions(X)
        baseline_score = scoring(y, baseline_pred)
        print(f"Baseline ROC-AUC: {baseline_score:.4f}")
        
        # Store importance for each feature and each repeat
        self.importances_ = np.zeros((n_features, self.n_repeats))
        
        for feat_idx in range(n_features):
            for repeat_idx in range(self.n_repeats):
                # Create a copy of X
                X_permuted = X.copy()
                
                # Shuffle the feature column
                np.random.shuffle(X_permuted[:, feat_idx])
                
                # Calculate score with permuted feature
                permuted_pred = self._get_predictions(X_permuted)
                permuted_score = scoring(y, permuted_pred)
                
                # Importance is the decrease in performance
                self.importances_[feat_idx, repeat_idx] = baseline_score - permuted_score
            
            if (feat_idx + 1) % 10 == 0:
                print(f"Processed {feat_idx + 1}/{n_features} features...")
        
        # Calculate mean and std
        self.importances_mean_ = np.mean(self.importances_, axis=1)
        self.importances_std_ = np.std(self.importances_, axis=1)
        
        # Create DataFrame sorted by importance
        results = pd.DataFrame({
            'feature': self.feature_names,
            'importance_mean': self.importances_mean_,
            'importance_std': self.importances_std_
        })
        
        # Sort by absolute importance (descending)
        results = results.reindex(
            results['importance_mean'].abs().sort_values(ascending=False).index
        )
        results = results.reset_index(drop=True)
        
        print(f"\nTop 10 Features by Importance:")
        for i, row in results.head(10).iterrows():
            print(f"  {i+1}. {row['feature']}: {row['importance_mean']:.6f} (±{row['importance_std']:.6f})")
        
        return results
    
    def save_results(self, 
                     results: pd.DataFrame, 
                     output_dir: str,
                     model_name: str = "model"):
        """
        Save importance results to CSV and generate visualization.
        
        Args:
            results: DataFrame from compute()
            output_dir: Directory to save outputs
            model_name: Model name for plot title
        """
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        # Save CSV
        csv_path = os.path.join(output_dir, 'feature_importance.csv')
        results.to_csv(csv_path, index=False, encoding='utf-8')
        print(f"\nFeature importance saved to: {csv_path}")
        
        # Generate visualization
        self.plot_importance(results, output_dir, model_name)
    
    def plot_importance(self, 
                        results: pd.DataFrame, 
                        output_dir: str,
                        model_name: str = "model",
                        top_n: int = 20):
        """
        Create horizontal bar chart of feature importance.
        
        Args:
            results: DataFrame from compute()
            output_dir: Directory to save plot
            model_name: Model name for title
            top_n: Number of top features to show
        """
        import os
        
        # Get top N features
        top_features = results.head(top_n).copy()
        
        # Reverse order for plotting (most important at top)
        top_features = top_features.iloc[::-1]
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Create horizontal bar chart
        y_pos = np.arange(len(top_features))
        bars = ax.barh(y_pos, 
                       top_features['importance_mean'].values,
                       xerr=top_features['importance_std'].values,
                       align='center',
                       color='steelblue',
                       alpha=0.8,
                       ecolor='gray',
                       capsize=3)
        
        # Customize appearance
        ax.set_yticks(y_pos)
        ax.set_yticklabels(top_features['feature'].values, fontsize=10)
        ax.set_xlabel('Permutation Importance (Decrease in ROC-AUC)', fontsize=11)
        ax.set_title(f'{model_name} - Feature Importance (Top {top_n})', fontsize=13)
        ax.grid(True, alpha=0.3, axis='x')
        
        # Add value labels
        for i, (bar, mean_val) in enumerate(zip(bars, top_features['importance_mean'].values)):
            if mean_val >= 0:
                ax.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height()/2,
                       f'{mean_val:.4f}', va='center', fontsize=8)
        
        plt.tight_layout()
        
        # Save figure
        png_path = os.path.join(output_dir, 'feature_importance.png')
        plt.savefig(png_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Feature importance plot saved to: {png_path}")


def compute_importance_for_model(model,
                                  X: np.ndarray,
                                  y: np.ndarray,
                                  feature_names: List[str],
                                  output_dir: str,
                                  model_name: str,
                                  n_repeats: int = 5,
                                  random_state: int = 42) -> pd.DataFrame:
    """
    Convenience function to compute and save feature importance.
    
    Args:
        model: Trained model
        X: Feature matrix
        y: Target labels
        feature_names: List of feature names
        output_dir: Directory to save outputs
        model_name: Model name for visualization
        n_repeats: Number of permutation repeats
        random_state: Random seed
        
    Returns:
        DataFrame with importance results
    """
    print(f"\n{'='*60}")
    print(f"Computing Permutation Importance for {model_name}")
    print(f"{'='*60}")
    print(f"Features: {len(feature_names)}, Samples: {len(X)}, Repeats: {n_repeats}")
    
    pi = PermutationImportance(
        model=model,
        feature_names=feature_names,
        n_repeats=n_repeats,
        random_state=random_state
    )
    
    results = pi.compute(X, y)
    pi.save_results(results, output_dir, model_name)
    
    return results
