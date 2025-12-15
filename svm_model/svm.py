"""
Support Vector Machine (SVM) Model Definition

This module implements a wrapper class for SVM classifier and Hinge Loss
for binary classification tasks.

References:
    [1] Cortes, C., & Vapnik, V. (1995).
        "Support-vector networks"
        Machine Learning, 20(3), 273-297.
        https://link.springer.com/article/10.1007/BF00994018

    [2] Boser, B. E., Guyon, I. M., & Vapnik, V. N. (1992).
        "A training algorithm for optimal margin classifiers"
        COLT '92. https://dl.acm.org/doi/10.1145/130385.130401

Author: Data Modeling Project
"""

from sklearn.svm import SVC
import numpy as np


class SVMModel:
    """
    SVM Model Wrapper Class.
    
    Wraps sklearn's SVC with a simplified interface for binary classification.
    Supports various kernel functions including RBF, linear, polynomial, and sigmoid.
    
    Args:
        C (float): Regularization parameter. Default: 1.0
        kernel (str): Kernel function ('linear', 'poly', 'rbf', 'sigmoid'). Default: 'rbf'
        gamma (str or float): Kernel coefficient. Default: 'scale'
        probability (bool): Whether to enable probability estimation. Default: True
        random_state (int): Random seed for reproducibility. Default: 42
        class_weight (str or dict): Class weight configuration. Default: 'balanced'
    """
    
    def __init__(self, C=1.0, kernel='rbf', gamma='scale',
                 probability=True, random_state=42, class_weight='balanced'):
        """
        Initialize SVM model.
        
        Args:
            C: Regularization parameter (inverse of regularization strength)
            kernel: Kernel function type ('linear', 'poly', 'rbf', 'sigmoid')
            gamma: Kernel coefficient for 'rbf', 'poly' and 'sigmoid'
            probability: Whether to enable probability estimates
            random_state: Random seed for reproducibility
            class_weight: Weights associated with classes
        """
        self.model = SVC(
            C=C,
            kernel=kernel,
            gamma=gamma,
            probability=probability,
            random_state=random_state,
            class_weight=class_weight
        )
    
    def fit(self, X, y):
        """Train the model."""
        self.model.fit(X, y)
        return self
    
    def predict(self, X):
        """Predict class labels."""
        return self.model.predict(X)
    
    def predict_proba(self, X):
        """Predict class probabilities."""
        return self.model.predict_proba(X)
    
    def get_support_vectors(self):
        """Get support vectors."""
        return self.model.support_vectors_
    
    def get_n_support(self):
        """Get number of support vectors for each class."""
        return self.model.n_support_


class HingeLoss:
    """
    Hinge Loss for SVM (Reference Implementation).
    
    Hinge loss is the standard loss function for Support Vector Machines.
    Note: sklearn's SVC handles this internally; this class is for reference only.
    
    Formula: L(y, f(x)) = max(0, 1 - y * f(x))
    
    Reference:
        Rosasco, L., et al. (2004).
        "Are Loss Functions All the Same?"
        Neural Computation, 16(5), 1063-1076.
    """
    
    def __call__(self, y_true, y_pred):
        """
        Compute Hinge Loss.
        
        Args:
            y_true: True labels (should be -1 or +1)
            y_pred: Predicted values (decision function output)
            
        Returns:
            float: Mean hinge loss value
        """
        # Convert labels to -1, +1 format
        y_true_converted = 2 * y_true - 1
        
        loss = np.maximum(0, 1 - y_true_converted * y_pred)
        return np.mean(loss)
