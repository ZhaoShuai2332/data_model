from sklearn.linear_model import LogisticRegression
import numpy as np

"""
Logistic Regression Model Definition

This module provides a lightweight wrapper around scikit-learn's `LogisticRegression` for
binary classification tasks on tabular data. The class exposes common helper
methods for model training, inference, and parameter inspection.

References:
    [1] Hosmer, D. W., Lemeshow, S., & Sturdivant, R. X. (2013). *Applied Logistic
        Regression* (3rd ed.). Wiley.
    [2] Pedregosa, F. *et al.* (2011). "Scikit-learn: Machine Learning in Python."
        *Journal of Machine Learning Research*, 12, 2825-2830.
"""

class LogisticModel:
    """Scikit-learn Logistic Regression wrapper."""
    
    def __init__(self, penalty='l2', C=1.0, solver='lbfgs', 
                 max_iter=1000, random_state=42, class_weight='balanced'):
        """
        Initialize Logistic Regression model
        
        Args:
            penalty: Regularization type ('l1', 'l2', 'elasticnet', None)
            C: Inverse of regularization strength
            solver: Optimization algorithm
            max_iter: Maximum number of iterations
            random_state: Random seed
            class_weight: Class weights
        """
        self.model = LogisticRegression(
            penalty=penalty,
            C=C,
            solver=solver,
            max_iter=max_iter,
            random_state=random_state,
            class_weight=class_weight
        )
    
    def fit(self, X, y):
        """Train the model"""
        self.model.fit(X, y)
        return self
    
    def predict(self, X):
        """Predict classes"""
        return self.model.predict(X)
    
    def predict_proba(self, X):
        """Predict probabilities"""
        return self.model.predict_proba(X)
    
    def get_coef(self):
        """Get model coefficients"""
        return self.model.coef_
    
    def get_intercept(self):
        """Get model intercept"""
        return self.model.intercept_


class WeightedBCELoss:
    """
    Weighted Binary Cross Entropy Loss (for class imbalance problems)
    Note: sklearn's LogisticRegression handles class imbalance internally, this class is for reference only
    """
    
    def __init__(self, pos_weight=1.0):
        self.pos_weight = pos_weight
    
    def __call__(self, y_true, y_pred_proba):
        """
        Calculate weighted BCE loss
        
        Args:
            y_true: True labels
            y_pred_proba: Predicted probabilities
        """
        epsilon = 1e-7
        y_pred_proba = np.clip(y_pred_proba, epsilon, 1 - epsilon)
        
        loss = -(self.pos_weight * y_true * np.log(y_pred_proba) + 
                 (1 - y_true) * np.log(1 - y_pred_proba))
        
        return np.mean(loss)
