import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from logistic import LogisticModel
from evaluate import MetricsCalculator
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import joblib
import os

"""
Code for model training and prediction
"""

class LogisticPredictor:
    """Logistic Regression Trainer with F1-score"""
    
    def __init__(self, penalty='l2', C=1.0, solver='lbfgs',
                 max_iter=1000, random_state=42, class_weight='balanced'):
        """
        Initialize the trainer
        
        Args:
            penalty: Regularization type
            C: Inverse of regularization strength
            solver: Optimization algorithm
            max_iter: Maximum number of iterations
            random_state: Random seed
            class_weight: Class weights
        """
        self.model = LogisticModel(
            penalty=penalty,
            C=C,
            solver=solver,
            max_iter=max_iter,
            random_state=random_state,
            class_weight=class_weight
        )
        
        self.metrics_calculator = MetricsCalculator()
        self.train_metrics = None
        self.val_metrics = None
    
    def fit(self, X_train, y_train, X_val=None, y_val=None):
        """
        Train the model
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
        """
        print(f"\nStarting Logistic Regression model training...")
        print(f"Training set: {len(X_train)}, Validation set: {len(X_val) if X_val is not None else 0}")
        
        # Train model
        self.model.fit(X_train, y_train)
        
        # Calculate training metrics
        train_pred = self.model.predict(X_train)
        train_pred_proba = self.model.predict_proba(X_train)[:, 1]
        self.train_metrics = self.metrics_calculator.calculate_metrics(
            y_train, train_pred, train_pred_proba
        )
        
        print("\nTraining set evaluation results:")
        self._print_metrics(self.train_metrics)
        
        # Calculate validation metrics
        if X_val is not None and y_val is not None:
            val_pred = self.model.predict(X_val)
            val_pred_proba = self.model.predict_proba(X_val)[:, 1]
            self.val_metrics = self.metrics_calculator.calculate_metrics(
                y_val, val_pred, val_pred_proba
            )
            
            print("\nValidation set evaluation results:")
            self._print_metrics(self.val_metrics)
            
            return {
                'train': self.train_metrics,
                'val': self.val_metrics
            }
        
        return {'train': self.train_metrics}
    
    def _print_metrics(self, metrics):
        """Print evaluation metrics"""
        print(f"  Accuracy:    {metrics['accuracy']:.4f}")
        print(f"  Precision:   {metrics['precision']:.4f}")
        print(f"  Recall:      {metrics['recall']:.4f}")
        print(f"  F1-Score:    {metrics['f1_score']:.4f}")
        print(f"  Specificity: {metrics['specificity']:.4f}")
        print(f"  ROC-AUC:     {metrics['roc_auc']:.4f}")
    
    def predict(self, X):
        """Predict class labels"""
        return self.model.predict(X)
    
    def predict_proba(self, X):
        """Predict probabilities"""
        return self.model.predict_proba(X)
    
    def save_model(self, filepath):
        """Save model"""
        joblib.dump(self.model.model, filepath)
        print(f"Model saved to: {filepath}")
    
    def load_model(self, filepath):
        """Load model"""
        self.model.model = joblib.load(filepath)
        return self
    
    def get_feature_importance(self, feature_names=None, top_n=20):
        """Get feature importance (absolute coefficient values)"""
        coef = self.model.get_coef().flatten()
        
        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(len(coef))]
        
        importance = list(zip(feature_names, coef))
        importance.sort(key=lambda x: abs(x[1]), reverse=True)
        
        return importance[:top_n]
    
    def plot_confusion_matrix(self, y_true, y_pred, save_path=None, title='Confusion Matrix'):
        """Plot confusion matrix"""
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(title)
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Confusion matrix saved to: {save_path}")
        plt.close()
    
    def plot_roc_curve(self, y_true_train, y_score_train, y_true_val=None, 
                       y_score_val=None, save_path=None):
        """Plot ROC curve"""
        plt.figure(figsize=(10, 8))
        
        # Training ROC
        fpr_train, tpr_train, _ = roc_curve(y_true_train, y_score_train)
        roc_auc_train = auc(fpr_train, tpr_train)
        plt.plot(fpr_train, tpr_train, 
                label=f'Train ROC (AUC = {roc_auc_train:.4f})', linewidth=2)
        
        # Validation ROC
        if y_true_val is not None and y_score_val is not None:
            fpr_val, tpr_val, _ = roc_curve(y_true_val, y_score_val)
            roc_auc_val = auc(fpr_val, tpr_val)
            plt.plot(fpr_val, tpr_val, 
                    label=f'Val ROC (AUC = {roc_auc_val:.4f})', linewidth=2)
        
        # Random Classifier
        plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Logistic Regression ROC Curve')
        plt.legend(loc='lower right')
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"ROC curve saved to: {save_path}")
        plt.close()
    
    def plot_training_history(self):
        """Plot training history comparison"""
        if self.train_metrics is None:
            print("Please train the model first")
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Metrics comparison bar chart
        metrics_names = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']
        train_values = [self.train_metrics[m] for m in metrics_names]
        
        x = np.arange(len(metrics_names))
        width = 0.35
        
        axes[0].bar(x - width/2, train_values, width, label='Train', alpha=0.8)
        
        if self.val_metrics is not None:
            val_values = [self.val_metrics[m] for m in metrics_names]
            axes[0].bar(x + width/2, val_values, width, label='Val', alpha=0.8)
        
        axes[0].set_ylabel('Score')
        axes[0].set_title('Training vs Validation Metrics')
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(metrics_names, rotation=45)
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        axes[0].set_ylim([0, 1])
        
        # Confusion matrix metrics
        if self.val_metrics is not None:
            metrics = self.val_metrics
            title = 'Validation Confusion Metrics'
        else:
            metrics = self.train_metrics
            title = 'Training Confusion Metrics'
        
        cm_values = [metrics['tp'], metrics['fp'], metrics['fn'], metrics['tn']]
        cm_labels = ['TP', 'FP', 'FN', 'TN']
        colors = ['green', 'orange', 'red', 'blue']
        
        axes[1].bar(cm_labels, cm_values, color=colors, alpha=0.7)
        axes[1].set_ylabel('Count')
        axes[1].set_title(title)
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('logistic_model\\outputs\\training_history.png', dpi=150)
        print("\nTraining history saved to: logistic_model\\outputs\\training_history.png")
        
        return fig
