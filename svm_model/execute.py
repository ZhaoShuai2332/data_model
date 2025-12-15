from data_process import DataPreprocessor
from train import SVMPredictor
from evaluate import MetricsCalculator
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_curve, auc, classification_report, confusion_matrix,
    precision_recall_curve, f1_score
)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
import joblib
import warnings
warnings.filterwarnings('ignore')

# Output logger class: outputs to both terminal and file
class TeeLogger:
    def __init__(self, filepath):
        self.terminal = sys.stdout
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        self.log = open(filepath, 'w', encoding='utf-8')
    
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()
    
    def flush(self):
        self.terminal.flush()
        self.log.flush()
    
    def close(self):
        self.log.close()

# Set up output logging
logger = TeeLogger('svm_model\\outputs\\res_print.txt')
sys.stdout = logger

SEED = 42
np.random.seed(SEED)


def main():
    """Main function"""
    
    print("=" * 60)
    print("High Risk Customer Prediction - SVM Model")
    print("=" * 60)
    
    # Ensure output directory exists
    os.makedirs('svm_model\\outputs', exist_ok=True)
    
    # 1. Data Loading
    preprocessor = DataPreprocessor()
    
    train_data = preprocessor.load_data('train.csv')
    test_data = preprocessor.load_data('test.csv')
    
    # 2. Data Preprocessing
    # Note: '是否高风险' is the column name in the CSV, keeping it as is for data access
    X, y = preprocessor.preprocess_train(train_data, target_col='是否高风险')
    
    # 3. Split Training and Validation Sets
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y
    )
    
    print(f"\nDataset Split:")
    print(f"Training set: {X_train.shape}, Positive ratio: {y_train.mean():.4f}")
    print(f"Validation set: {X_val.shape}, Positive ratio: {y_val.mean():.4f}")
    
    # SVM is slow on large datasets, use sampling for acceleration
    MAX_TRAIN_SAMPLES = 32000  # Limit training samples
    if len(X_train) > MAX_TRAIN_SAMPLES:
        print(f"\n[Note] SVM training sample sampling: {len(X_train)} -> {MAX_TRAIN_SAMPLES}")
        # Stratified sampling to maintain class ratio
        from sklearn.model_selection import train_test_split as stratified_sample
        X_train, _, y_train, _ = stratified_sample(
            X_train, y_train, 
            train_size=MAX_TRAIN_SAMPLES, 
            random_state=SEED, 
            stratify=y_train
        )
        print(f"Training set after sampling: {X_train.shape}, Positive ratio: {y_train.mean():.4f}")
    
    # 4. Create and Train Model
    predictor = SVMPredictor(
        C=1.0,
        kernel='rbf',
        gamma='scale',
        probability=True,
        random_state=SEED,
        class_weight='balanced'
    )
    
    # Train
    history = predictor.fit(X_train, y_train, X_val, y_val)
    
    # 5. Detailed Evaluation on Validation Set
    print("\n" + "=" * 60)
    print("Validation Set Detailed Evaluation:")
    print("=" * 60)
    
    val_probs = predictor.predict_proba(X_val)[:, 1]
    
    # Find best threshold
    print("\nFinding optimal classification threshold...")
    best_threshold_f1, best_f1_at_threshold = MetricsCalculator.find_best_threshold(
        y_val, val_probs, metric='f1'
    )
    print(f"Best F1 Threshold: {best_threshold_f1:.4f}, F1-Score: {best_f1_at_threshold:.4f}")
    
    # Evaluate using default threshold 0.5
    val_preds_05 = (val_probs >= 0.5).astype(int)
    metrics_05 = MetricsCalculator.calculate_metrics(y_val, val_preds_05, val_probs, 0.5)
    
    print(f"\nUsing Threshold 0.5:")
    print(f"  Accuracy:    {metrics_05['accuracy']:.4f}")
    print(f"  Precision:   {metrics_05['precision']:.4f}")
    print(f"  Recall:      {metrics_05['recall']:.4f}")
    print(f"  F1-Score:    {metrics_05['f1_score']:.4f}")
    print(f"  Specificity: {metrics_05['specificity']:.4f}")
    print(f"  ROC AUC:     {metrics_05['roc_auc']:.4f}")
    print(f"  PR AUC:      {metrics_05['pr_auc']:.4f}")
    
    # Evaluate using best threshold
    val_preds_best = (val_probs >= best_threshold_f1).astype(int)
    metrics_best = MetricsCalculator.calculate_metrics(y_val, val_preds_best, val_probs, best_threshold_f1)
    
    print(f"\nUsing Best Threshold {best_threshold_f1:.4f}:")
    print(f"  Accuracy:    {metrics_best['accuracy']:.4f}")
    print(f"  Precision:   {metrics_best['precision']:.4f}")
    print(f"  Recall:      {metrics_best['recall']:.4f}")
    print(f"  F1-Score:    {metrics_best['f1_score']:.4f}")
    print(f"  Specificity: {metrics_best['specificity']:.4f}")
    
    # Confusion Matrix
    print("\nConfusion Matrix (Threshold=0.5):")
    cm = confusion_matrix(y_val, val_preds_05)
    print(cm)
    print(f"\n  TN: {metrics_05['tn']}, FP: {metrics_05['fp']}")
    print(f"  FN: {metrics_05['fn']}, TP: {metrics_05['tp']}")
    
    # Classification Report
    print("\nClassification Report (Threshold=0.5):")
    print(classification_report(y_val, val_preds_05, target_names=['Low Risk', 'High Risk']))
    
    # 6. Plot Evaluation Charts
    # Calculate training set prediction probabilities
    train_probs = predictor.predict_proba(X_train)[:, 1]
    
    # 1. Plot ROC Curve (Train + Val)
    fpr_train, tpr_train, _ = roc_curve(y_train, train_probs)
    fpr_val, tpr_val, _ = roc_curve(y_val, val_probs)
    roc_auc_train = auc(fpr_train, tpr_train)
    roc_auc_val = auc(fpr_val, tpr_val)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr_train, tpr_train, lw=2, alpha=0.8, label=f'Train (AUC = {roc_auc_train:.4f})')
    plt.plot(fpr_val, tpr_val, lw=2, alpha=0.8, label=f'Val (AUC = {roc_auc_val:.4f})')
    plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('SVM ROC Curve')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.savefig('svm_model\\outputs\\roc_curve.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("\nROC curve saved: svm_model\\outputs\\roc_curve.png")
    
    # 2. Plot PR Curve
    precision_curve, recall_curve, _ = precision_recall_curve(y_val, val_probs)
    pr_auc = auc(recall_curve, precision_curve)
    
    plt.figure(figsize=(8, 6))
    plt.plot(recall_curve, precision_curve, lw=2, label=f'SVM (AUC = {pr_auc:.4f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="lower left")
    plt.grid(True, alpha=0.3)
    plt.savefig('svm_model\\outputs\\pr_curve.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("PR curve saved: svm_model\\outputs\\pr_curve.png")
    
    # 3. F1-score vs Threshold
    thresholds = np.linspace(0, 1, 100)
    f1_scores = []
    for t in thresholds:
        preds = (val_probs >= t).astype(int)
        f1_scores.append(f1_score(y_val, preds, zero_division=0))
    
    plt.figure(figsize=(8, 6))
    plt.plot(thresholds, f1_scores, lw=2)
    plt.axvline(x=best_threshold_f1, color='r', linestyle='--', 
                label=f'Best Threshold = {best_threshold_f1:.3f}')
    plt.xlabel('Threshold')
    plt.ylabel('F1-Score')
    plt.title('F1-Score vs Threshold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('svm_model\\outputs\\f1_threshold.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("F1 threshold plot saved: svm_model\\outputs\\f1_threshold.png")
    
    # 4. Confusion Matrix Visualization
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(2)
    plt.xticks(tick_marks, ['Low Risk', 'High Risk'])
    plt.yticks(tick_marks, ['Low Risk', 'High Risk'])
    
    for i in range(2):
        for j in range(2):
            plt.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig('svm_model\\outputs\\confusion_matrix.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Confusion matrix saved: svm_model\\outputs\\confusion_matrix.png")
    
    # 7. Plot Training History
    predictor.plot_training_history()
    
    # 8. Predict on Test Set
    print("\n" + "=" * 60)
    print("Test Set Prediction:")
    print("=" * 60)
    
    X_test = preprocessor.preprocess_test(test_data)
    test_probs = predictor.predict_proba(X_test)[:, 1]
    
    print(f"Prediction complete, number of samples: {len(test_probs)}")
    print(f"Predicted high risk ratio (threshold=0.5): {(test_probs > 0.5).mean():.4f}")
    print(f"Predicted high risk ratio (best threshold={best_threshold_f1:.3f}): {(test_probs > best_threshold_f1).mean():.4f}")
    print(f"Predicted probability range: [{test_probs.min():.4f}, {test_probs.max():.4f}]")
    
    # 9. Save Prediction Results
    result_df = pd.DataFrame(test_probs)
    result_df.to_csv('svm_model\\outputs\\result.csv', 
                     encoding='utf-8', 
                     index=False, 
                     header=False)
    
    print("\nPrediction results saved: svm_model\\outputs\\result.csv")
    print(f"File format: {len(test_probs)} rows × 1 column")
    
    # 10. Save Model
    predictor.save_model('svm_model\\outputs\\svm_model_final.pkl')
    
    # Save Preprocessor
    joblib.dump(preprocessor, 'svm_model\\outputs\\preprocessor.pkl')
    print("Preprocessor saved: svm_model\\outputs\\preprocessor.pkl")
    
    # 11. Print Final Summary
    print("\n" + "=" * 60)
    print("Final Summary:")
    print("=" * 60)
    print(f"Best ROC AUC:     {history['val']['roc_auc']:.4f}")
    print(f"Best F1-Score:    {best_f1_at_threshold:.4f} (Threshold={best_threshold_f1:.3f})")
    print(f"Validation Precision: {metrics_best['precision']:.4f}")
    print(f"Validation Recall:    {metrics_best['recall']:.4f}")
    print("=" * 60)
    print("\nAll tasks completed!")


if __name__ == '__main__':
    main()
